//! Polyhedral template matching (design §14).
//!
//! The algorithm, as published in Larsen & Schiøtz, * modelling atomic systems
//! with polyhedral template matching* (Modelling Simul. Mater. Sci. Eng.
//! 24:055007, 2016), and as implemented in OVITO's bundled `lib_ptm`:
//!
//! 1. Collect up to 18 neighbours from the host's neighbour list, already in
//!    ascending distance order, with the central atom at `points[0]`.
//! 2. Take the first `num_nbrs` of that ordering as the point set to classify.
//! 3. **Normalise** it: translate the centroid to the origin and rescale so the
//!    mean squared neighbour distance is 1.
//! 4. Enumerate the **convex hull**, canonicalise it into a graph, and look the
//!    resulting graph up in the structure's table of known hull graphs.
//! 5. For every automorphism of a matching graph, fit a rotation (QCP) and keep
//!    the lowest RMSD.
//! 6. Report the best structure whose normalised RMSD is under the cutoff;
//!    everything else is `OTHER`.
//!
//! Step 1 is worth being precise about: upstream `ptm_index` asks the host for
//! up to [`MAX_INPUT_POINTS`] already-ordered neighbours. [`Ptm::compute`]
//! passes them by distance. OVITO 3.15 passes them in `ptm_preorder_neighbours`
//! order (decreasing Voronoi solid angle over the 18 nearest), which
//! [`Ptm::compute_ordered`] with [`NeighbourOrder::Voronoi`] reproduces: OVITO's
//! types and RMSDs atom for atom on the reference snapshots (design §14.1).
//!
//! ## Shipped scope
//!
//! All eight PTM structures: FCC, HCP, BCC, ICO and SC from the first
//! neighbour shell, and cubic diamond, hexagonal diamond and graphene from the
//! two-shell environments upstream builds with
//! `calculate_two_shell_neighbour_ordering` ([`two_shell_environment`]). The
//! default check set is OVITO's (FCC, HCP, BCC); the others are opt-in, as they
//! are in OVITO. RMSD, scale, interatomic distance, lattice constant and
//! orientation are always written; the alloy ordering type, the elastic
//! deformation gradient and the template index are opt-in ([`output`],
//! [`Ptm::run_extended`], module [`remap`]). All of them match OVITO.
//!
//! ## Layout
//!
//! * [`tables`] — generated ideal templates, hull graphs and automorphisms.
//! * [`polar`] — inner product, optimal rotation, RMSD (QCP).
//! * [`voronoi`] — neighbour ordering by Voronoi solid angle.
//! * [`hull`] — incremental convex hull and facet winding.
//! * [`graph`] — hull degree and coloured canonical form.
//! * [`matcher`] — template matching against the hull-graph tables.
//! * [`remap`] — template remap, deformation gradient, alloy ordering (opt-in).

use crate::analysis::{Analysis, AnalysisContext, Summary};
use crate::data::property::PropertyStore;
use crate::error::Result;
use crate::prof;

pub mod graph;
pub mod hull;
pub mod matcher;
pub mod nbrlist;
pub mod normalize;
pub mod polar;
pub mod quat;
pub mod quat_tables;
pub mod refdata;
pub mod remap;
pub mod remap_tables;
pub mod tables;
pub mod voronoi;

/// Maximum number of neighbours PTM will consider (upstream `PTM_MAX_NBRS`).
pub const MAX_NBRS: usize = 16;
/// Maximum point set size: the central atom plus its neighbours.
pub const MAX_POINTS: usize = MAX_NBRS + 1;
/// Maximum hull facets (`PTM_MAX_FACETS`).
pub const MAX_FACETS: usize = 28;
/// Maximum hull edges (`PTM_MAX_EDGES`).
pub const MAX_EDGES: usize = 42;

/// Structure identification results, matching OVITO's `PTM_MATCH_*` codes.
///
/// The numbering is OVITO's, and deliberately *not* the same as this crate's
/// CNA codes: PTM has a value for SC that CNA's structure codes leave unused,
/// and the two analyses are stored in different properties.
pub mod structure {
    pub const OTHER: i32 = 0;
    pub const FCC: i32 = 1;
    pub const HCP: i32 = 2;
    pub const BCC: i32 = 3;
    pub const ICO: i32 = 4;
    pub const SC: i32 = 5;
    /// Cubic diamond (`PTM_MATCH_DCUB`).
    pub const DCUB: i32 = 6;
    /// Hexagonal diamond (`PTM_MATCH_DHEX`).
    pub const DHEX: i32 = 7;
    /// Graphene (`PTM_MATCH_GRAPHENE`).
    pub const GRAPHENE: i32 = 8;
    /// Double HCP (ABAC), derived from PTM's FCC and HCP labels by
    /// [`crate::analysis::stacking`]; PTM itself never produces it, so its
    /// own output stays comparable with OVITO's.
    pub const DHCP: i32 = 32;
}

/// Which structures to test for, as a bitmask.
pub mod check {
    pub const FCC: u32 = 1 << 0;
    pub const HCP: u32 = 1 << 1;
    pub const BCC: u32 = 1 << 2;
    pub const ICO: u32 = 1 << 3;
    pub const SC: u32 = 1 << 4;
    pub const DCUB: u32 = 1 << 5;
    pub const DHEX: u32 = 1 << 6;
    pub const GRAPHENE: u32 = 1 << 7;
    /// Every structure (`PTM_CHECK_ALL`).
    pub const ALL: u32 = FCC | HCP | BCC | ICO | SC | DCUB | DHEX | GRAPHENE;
    /// The structures matched from the first neighbour shell.
    pub const FIRST_SHELL: u32 = FCC | HCP | BCC | ICO | SC;
    /// The structures matched from a two-shell environment.
    pub const TWO_SHELL: u32 = DCUB | DHEX | GRAPHENE;

    /// The `lib_ptm` C++ library's own default (`PTM_CHECK_DEFAULT`).
    ///
    /// Note this is *not* what OVITO runs. The OVITO PTM modifier's default
    /// structure list is
    /// [`OVITO_DEFAULT`](Self::OVITO_DEFAULT).
    pub const DEFAULT: u32 = FCC | HCP | ICO | BCC;

    /// The set OVITO's PTM modifier checks by default: FCC, HCP and BCC.
    ///
    /// OVITO 3.15 ships ICO, SC, both diamonds and graphene disabled
    /// (`PolyhedralTemplateMatchingModifier().structures`: checked against the
    /// installed 3.15.0), and enabling extra structures changes results,
    /// because the lowest RMSD wins. The captured PTM oracles were recorded
    /// with this set.
    pub const OVITO_DEFAULT: u32 = FCC | HCP | BCC;
}

/// Opt-in per-atom outputs of [`Ptm::run_extended`], as a bitmask. None is
/// written by default, so default outputs and the compact [`PtmAtom`] are
/// unchanged. Each is what OVITO 3.15 writes for its matching option, and,
/// like OVITO, each describes the best fit even when the RMSD cutoff rejects
/// the atom (zero when nothing matched at all).
pub mod output {
    /// `StructureOrdering` (u8, [`super::remap::ordering`] codes): OVITO's
    /// "Ordering Type" (`output_ordering`), from the particle types.
    pub const ORDERING: u32 = 1 << 0;
    /// `StructureDeformationGradient` (9 × f64, row-major `F[3 i + j]` =
    /// F_ij): OVITO's "Elastic Deformation Gradient"
    /// (`output_deformation_gradient`).
    pub const DEFORMATION_GRADIENT: u32 = 1 << 1;
    /// `StructureTemplate` (u8): upstream's `best_template_index`, which of a
    /// structure's alternative templates the conventional orientation refers
    /// to (0 for SC, FCC, BCC, ICO; 0–1 for HCP, DCUB, graphene; 0–3 DHEX).
    pub const TEMPLATE: u32 = 1 << 2;
    /// `Correspondences` (i64): OVITO's "Correspondences" (written with
    /// `output_orientation`), upstream `ptm_encode_correspondences` of the
    /// template-to-neighbour mapping and the template index
    /// ([`super::remap::encode_correspondences`]).
    pub const CORRESPONDENCES: u32 = 1 << 3;
    /// `StructureDeformationResidual` (3 × f64): upstream's `F_res`, the
    /// per-axis squared residual of the F fit (PTM library as bundled with
    /// LAMMPS; OVITO 3.15 has no such output).
    pub const RESIDUAL: u32 = 1 << 4;
    /// `StructurePolarU`, `StructurePolarP` (9 × f64 each, row-major): the
    /// upstream polar decomposition `F = P U` (rotation U, symmetric left
    /// stretch P), as the LAMMPS-bundled library's `output_data`; not in
    /// OVITO 3.15.
    pub const POLAR: u32 = 1 << 5;
    /// Every opt-in output.
    pub const ALL: u32 =
        ORDERING | DEFORMATION_GRADIENT | TEMPLATE | CORRESPONDENCES | RESIDUAL | POLAR;

    /// Parse one `--ptm-outputs` / `outputs=` name.
    pub fn from_name(name: &str) -> Option<u32> {
        Some(match name {
            "ordering" => ORDERING,
            "F" | "f" | "deformation_gradient" => DEFORMATION_GRADIENT,
            "template" => TEMPLATE,
            "correspondences" => CORRESPONDENCES,
            "residual" | "F_res" | "f_res" => RESIDUAL,
            "polar" | "U" | "P" => POLAR,
            "all" => ALL,
            _ => return None,
        })
    }
}

/// Neighbour counts per structure (upstream `PTM_NUM_NBRS_*`), as point counts
/// including the central atom.
pub mod num_points {
    /// `PTM_NUM_NBRS_SC + 1`
    pub const SC: usize = 6 + 1;
    /// `PTM_NUM_NBRS_FCC + 1`
    pub const FCC: usize = 12 + 1;
    /// `PTM_NUM_NBRS_HCP + 1`
    pub const HCP: usize = 12 + 1;
    /// `PTM_NUM_NBRS_ICO + 1`
    pub const ICO: usize = 12 + 1;
    /// `PTM_NUM_NBRS_BCC + 1`
    pub const BCC: usize = 14 + 1;
    /// `PTM_NUM_NBRS_DCUB + 1` (4 inner + 12 outer)
    pub const DCUB: usize = 16 + 1;
    /// `PTM_NUM_NBRS_DHEX + 1`
    pub const DHEX: usize = 16 + 1;
    /// `PTM_NUM_NBRS_GRAPHENE + 1` (3 inner + 6 outer)
    pub const GRAPHENE: usize = 9 + 1;
    /// Widest reference, used as a compile-time bound.
    pub const WIDEST: usize = {
        let mut w = SC;
        if FCC > w {
            w = FCC;
        }
        if HCP > w {
            w = HCP;
        }
        if ICO > w {
            w = ICO;
        }
        if BCC > w {
            w = BCC;
        }
        w
    };
}

pub struct Ptm {
    /// RMSD above which a structural match is rejected and the atom becomes
    /// [`structure::OTHER`].
    ///
    /// The default is OVITO's, and it is in units of the mean neighbour length,
    /// because the point set is normalised to mean length 1 before the fit: 0.1
    /// means "within 10 % of ideal". Note that `lib_ptm` itself has no cutoff
    /// at all; this is the value the host applies to the RMSD it gets back.
    pub rmsd_cutoff: f64,
    /// Which structures to test, as a bitmask of [`check`] flags.
    pub flags: u32,
}

impl Default for Ptm {
    fn default() -> Self {
        Self {
            rmsd_cutoff: 0.1,
            flags: check::OVITO_DEFAULT,
        }
    }
}

/// Clock read for phase timing that costs one branch when profiling is off.
///
/// `Instant::now()` is a vDSO call, not a syscall, but it is not free, and at
/// 128M atoms it would be six calls per atom. With `--profile` absent all of
/// this collapses to one relaxed atomic load per phase.
#[inline]
fn mark() -> Option<std::time::Instant> {
    if prof::profiling_enabled() {
        Some(std::time::Instant::now())
    } else {
        None
    }
}

/// Fold the time since `t` into the named thread-local phase counter.
#[inline]
fn mark_end(name: &'static str, t: Option<std::time::Instant>) {
    if let Some(t) = t {
        prof::accumulate(name, t.elapsed().as_secs_f64());
    }
}

impl Ptm {
    /// Classify a single atom, mirroring upstream `ptm_index`.
    ///
    /// `res` is reset to the failure case first, so a caller can pass it in
    /// reused. `ch` is scratch; nothing in it carries over between calls.
    pub fn classify(
        &self,
        cell: &crate::neighbor::cell_list::CellList,
        i: usize,
        nb: &mut nbrlist::Neighbourhood,
        ch: &mut hull::ConvexHull,
        res: &mut matcher::MatchResult,
    ) -> Result<()> {
        *res = matcher::MatchResult::new(self.rmsd_cutoff);
        if self.flags == 0 {
            return Ok(());
        }
        let t_gather = mark();
        *nb = nbrlist::Neighbourhood::gather(cell, i)?;
        mark_end("ptm.gather", t_gather);
        self.classify_gathered(nb, ch, res);
        Ok(())
    }

    /// Classify an atom whose neighbourhood has already been gathered.
    ///
    /// This is the whole of `ptm_index` after the neighbour query, so any
    /// neighbour source (the cell list, the k-NN index) feeds the same match.
    pub fn classify_gathered(
        &self,
        nb: &nbrlist::Neighbourhood,
        ch: &mut hull::ConvexHull,
        res: &mut matcher::MatchResult,
    ) {
        *res = matcher::MatchResult::new(self.rmsd_cutoff);
        if self.flags == 0 {
            return;
        }
        self.match_first_shell(nb, ch, res);
        self.finish(res);
    }

    /// The first-shell matches (SC, FCC/HCP/ICO, BCC) of `ptm_index`, leaving
    /// the best fit in `res` without applying the cutoff.
    fn match_first_shell(
        &self,
        nb: &nbrlist::Neighbourhood,
        ch: &mut hull::ConvexHull,
        res: &mut matcher::MatchResult,
    ) {
        use nbrlist::MAX_INPUT_POINTS;
        use normalize::normalize_vertices_n;

        if self.flags & check::FIRST_SHELL == 0 {
            return;
        }

        // Upstream picks the minimum point count with a *cascade* of
        // overwriting assignments, not a maximum, so the last group enabled
        // wins. Reproduced verbatim: with the default flags (which include
        // BCC) the minimum is 15 even for an SC check, so a 13-neighbour
        // environment is rejected before SC is ever tried.
        let mut min_points = num_points::SC;
        if self.flags & (check::FCC | check::HCP | check::ICO) != 0 {
            min_points = num_points::FCC;
        }
        if self.flags & check::BCC != 0 {
            min_points = num_points::BCC;
        }

        if nb.len < min_points {
            // Upstream returns -1 here, which its callers treat as "no match"
            // for the first-shell structures.
            return;
        }

        // Upstream normalises every point the host returned, not just the
        // prefix any single reference reads. The QCP fit absorbs the scale
        // either way, but matching the count keeps the hull input identical.
        let t_norm = mark();
        let (ch_points, _) = normalize_vertices_n::<MAX_INPUT_POINTS>(&nb.points, nb.len);
        ch.ok = false;
        mark_end("ptm.normalize", t_norm);

        let t_match = mark();

        if self.flags & check::SC != 0 {
            // A hull error is a per-atom "no match", not a frame failure: the
            // matcher already swallows degenerate and over-faceted hulls and
            // returns `Ok`, so this arm is defensive. `res` is still the
            // failure case here, so leaving it alone is the correct result.
            let _ = matcher::match_general::<MAX_INPUT_POINTS>(
                &refdata::SC,
                &ch_points,
                &nb.points,
                ch,
                res,
            );
        }
        if self.flags & (check::FCC | check::HCP | check::ICO) != 0 {
            let _ = matcher::match_fcc_hcp_ico::<MAX_INPUT_POINTS>(
                &ch_points, &nb.points, self.flags, ch, res,
            );
        }
        if self.flags & check::BCC != 0 {
            let _ = matcher::match_general::<MAX_INPUT_POINTS>(
                &refdata::BCC,
                &ch_points,
                &nb.points,
                ch,
                res,
            );
        }
        mark_end("ptm.match", t_match);
    }

    /// The two-shell matches of `ptm_index`: cubic and hexagonal diamond from
    /// `dmn` (4 + 12 neighbours), graphene from `grp` (3 + 6). `None` is an
    /// environment that could not be built, which upstream also skips.
    fn match_two_shell(
        &self,
        dmn: Option<&[[f64; 3]; MAX_POINTS]>,
        grp: Option<&[[f64; 3]; MAX_POINTS]>,
        ch: &mut hull::ConvexHull,
        res: &mut matcher::MatchResult,
    ) {
        if self.flags & (check::DCUB | check::DHEX) != 0 {
            if let Some(points) = dmn {
                let (ch_points, _) =
                    normalize::normalize_vertices_n::<MAX_POINTS>(points, num_points::DCUB);
                ch.ok = false;
                matcher::match_dcub_dhex(&ch_points, points, self.flags, ch, res);
            }
        }
        if self.flags & check::GRAPHENE != 0 {
            if let Some(points) = grp {
                matcher::match_graphene(points, res);
            }
        }
    }

    /// Derived geometry, orientation and the host's cutoff for the best fit.
    fn finish(&self, res: &mut matcher::MatchResult) {
        // Upstream applies *no* cutoff: `ptm_index` always returns the
        // best-fitting structure by RMSD and leaves the accept/reject decision
        // to the caller, which is why the C++ library has no `rmsd_cutoff`
        // anywhere in it. OVITO then compares `rmsd` against its own
        // `rmsd_cutoff` (0.1 by default) and reports Other if it loses. The
        // decision has to happen here, or the cutoff is silently ignored and
        // every environment is reported as its best-fitting structure.
        //
        // `rmsd`, the derived geometry and the orientation are deliberately
        // left alone: they describe the best fit that was found, which is what
        // OVITO's RMSD/interatomic-distance/lattice-constant outputs report,
        // and they are the interesting numbers when a structure is rejected.
        // Only the *label* is withdrawn. This is why an atom can be
        // `OTHER` and still carry a finite RMSD, and why the reference
        // snapshots have 27,112 such atoms on Al rather than 34,933.
        if res.best_kind != structure::OTHER {
            res.interatomic_distance = matcher::interatomic_distance(res.best_kind, res.scale);
            res.lattice_constant =
                matcher::lattice_constant(res.best_kind, res.interatomic_distance);
            // Reduce the fit's quaternion into the fundamental zone of the
            // structure it was fitted to, so the result is a canonical
            // representative rather than whichever member of the symmetry coset
            // this particular neighbour permutation happened to select. Grain
            // segmentation (design §15) cannot group orientations without this.
            res.orientation = res.q;
            quat::rotate_into_structure_fundamental_zone(res.best_kind, &mut res.orientation);
        }
        if !res.is_match() {
            res.kind = structure::OTHER;
        }
    }
}

/// Upstream `calculate_two_shell_neighbour_ordering`: the central atom, its
/// first `num_inner` neighbours, then `num_outer` neighbours of each inner
/// one, laid out as `points[1 + num_inner + num_outer * k + c]`.
///
/// `central` is the central atom's ordered list (without itself) and
/// `inner[k]` the list of `central[k]`, each in the host's order. The outer
/// candidates are pooled over all inner atoms and taken by rank in their
/// inner atom's list (upstream sorts the pool by rank), skipping atoms already
/// taken (the centre, the inner atoms, earlier picks), until each inner atom
/// has `num_outer`. Atoms are identified by index, as upstream's
/// `unordered_set`, not by periodic image. Ties in rank are resolved by inner
/// atom (a stable sort); upstream's `std::sort` is not stable, which only
/// matters when one atom is a candidate of two inner atoms at the same rank.
///
/// `None` when the environment cannot be completed (upstream returns -1).
pub fn two_shell_environment(
    num_inner: usize,
    num_outer: usize,
    centre: u32,
    central: &[(u32, [f64; 3], f64)],
    inner: &[Vec<(u32, [f64; 3], f64)>],
) -> Option<[[f64; 3]; MAX_POINTS]> {
    two_shell_environment_atoms(num_inner, num_outer, centre, central, inner).map(|(p, _)| p)
}

/// [`two_shell_environment`] with the atom index of every point (the
/// centre at 0), in the same layout.
pub fn two_shell_environment_atoms(
    num_inner: usize,
    num_outer: usize,
    centre: u32,
    central: &[(u32, [f64; 3], f64)],
    inner: &[Vec<(u32, [f64; 3], f64)>],
) -> Option<([[f64; 3]; MAX_POINTS], [u32; MAX_POINTS])> {
    debug_assert!(num_inner <= 4 && num_inner * (1 + num_outer) < MAX_POINTS);
    // Upstream counts the atom itself, so `num_input_points` is `len + 1`.
    if central.len() < num_inner || inner.len() < num_inner {
        return None;
    }
    let mut points = [[0.0f64; 3]; MAX_POINTS];
    // Points are taken in layout order, so `taken[slot]` is not the slot's
    // atom; `atoms` records it.
    let mut atoms = [u32::MAX; MAX_POINTS];
    atoms[0] = centre;
    let mut taken = [u32::MAX; MAX_POINTS];
    let mut n_taken = 0usize;
    taken[n_taken] = centre;
    n_taken += 1;
    for k in 0..num_inner {
        points[1 + k] = central[k].1;
        atoms[1 + k] = central[k].0;
        taken[n_taken] = central[k].0;
        n_taken += 1;
    }

    // (rank, inner, atom, offset from the centre)
    const POOL: usize = 4 * nbrlist::MAX_INPUT_POINTS;
    let mut pool = [(0u32, 0u8, 0u32, [0.0f64; 3]); POOL];
    let mut n_pool = 0usize;
    for k in 0..num_inner {
        let list = &inner[k];
        if list.len() < num_inner {
            return None;
        }
        let base = central[k].1;
        // Rank 0 is the inner atom itself, always taken.
        for (r, &(atom, dr, _)) in list.iter().enumerate() {
            if taken[..n_taken].contains(&atom) {
                continue;
            }
            pool[n_pool] = (
                r as u32 + 1,
                k as u8,
                atom,
                [dr[0] + base[0], dr[1] + base[1], dr[2] + base[2]],
            );
            n_pool += 1;
        }
    }
    pool[..n_pool].sort_by_key(|e| e.0);

    let mut counts = [0usize; 4];
    let mut found = 0usize;
    for &(_, k, atom, off) in &pool[..n_pool] {
        let k = k as usize;
        if counts[k] >= num_outer || taken[..n_taken].contains(&atom) {
            continue;
        }
        points[1 + num_inner + num_outer * k + counts[k]] = off;
        atoms[1 + num_inner + num_outer * k + counts[k]] = atom;
        taken[n_taken] = atom;
        n_taken += 1;
        counts[k] += 1;
        found += 1;
        if found >= num_inner * num_outer {
            break;
        }
    }
    (found == num_inner * num_outer).then_some((points, atoms))
}

/// Compact per-atom PTM result for consumers that hold it for a whole frame
/// (segmentation, defects): 28 B/atom instead of the 72 B `AtomResult` plus
/// f64 property columns.
#[derive(Clone, Copy, Debug, Default)]
pub struct PtmAtom {
    /// Accepted structure ([`structure`] code), `OTHER` when rejected.
    pub structure: u8,
    pub rmsd: f32,
    /// First-neighbour distance of the best fit.
    pub interatomic_distance: f32,
    /// Fundamental-zone orientation `(w, x, y, z)` of the best fit.
    pub orientation: [f32; 4],
}

impl PtmAtom {
    #[inline]
    pub fn structure(&self) -> i32 {
        self.structure as i32
    }
    #[inline]
    pub fn q(&self) -> [f64; 4] {
        self.orientation.map(f64::from)
    }
}

/// The point set the winning structure was matched on (upstream's `env`):
/// what the opt-in outputs read.
pub(crate) struct Env<'a> {
    /// The environment's points, centre first, in matcher order.
    points: &'a [[f64; 3]],
    atoms: EnvAtoms<'a>,
}

/// Distance rank (0-based, in the k-NN list as returned before any Voronoi
/// reordering) of each entry of an ordered neighbour list: what upstream's
/// `env->correspondences` record, minus one.
type Ranks<'a> = &'a [u8];

enum EnvAtoms<'a> {
    /// First-shell: the centre, then the host's ordered neighbour list.
    First {
        centre: u32,
        nbrs: &'a [(u32, [f64; 3], f64)],
        ranks: Ranks<'a>,
    },
    /// Two-shell: the atom of every layout slot, plus the lists the slots
    /// were drawn from (the centre's, then each inner atom's).
    Two {
        atoms: &'a [u32; MAX_POINTS],
        nbrs: &'a [(u32, [f64; 3], f64)],
        ranks: Ranks<'a>,
        inner: &'a [Vec<(u32, [f64; 3], f64)>],
        inner_ranks: &'a [Vec<u8>],
    },
}

impl<'a> Env<'a> {
    fn first(
        centre: usize,
        nb: &'a nbrlist::Neighbourhood,
        nbrs: &'a [(u32, [f64; 3], f64)],
        ranks: Ranks<'a>,
    ) -> Self {
        Env {
            points: &nb.points[..nb.len],
            atoms: EnvAtoms::First {
                centre: centre as u32,
                nbrs,
                ranks,
            },
        }
    }

    /// Atom index of point `j`.
    fn atom(&self, j: usize) -> u32 {
        match self.atoms {
            EnvAtoms::First { centre, nbrs, .. } => {
                if j == 0 {
                    centre
                } else {
                    nbrs[j - 1].0
                }
            }
            EnvAtoms::Two { atoms, .. } => atoms[j],
        }
    }

    /// Upstream's `env->correspondences[j]` for structure `kind`: 0 for the
    /// centre, else one plus the point's distance rank in the neighbour list
    /// it was taken from (the centre's for first-shell points and inner
    /// atoms, its inner atom's for an outer point).
    fn correspondence(&self, kind: i32, j: usize) -> u8 {
        if j == 0 {
            return 0;
        }
        match self.atoms {
            EnvAtoms::First { ranks, .. } => ranks[j - 1] + 1,
            EnvAtoms::Two {
                atoms,
                nbrs,
                ranks,
                inner,
                inner_ranks,
            } => {
                let (n_inner, n_outer) = if kind == structure::GRAPHENE {
                    (3, 2)
                } else {
                    (4, 3)
                };
                if j <= n_inner {
                    return ranks[j - 1] + 1;
                }
                let k = (j - 1 - n_inner) / n_outer;
                let base = nbrs[k].1;
                // The slot holds `offset + base` of one entry of inner atom
                // `k`'s list, computed exactly so (`two_shell_environment`).
                let p = self.points[j];
                inner[k]
                    .iter()
                    .position(|&(a, d, _)| {
                        a == atoms[j]
                            && d[0] + base[0] == p[0]
                            && d[1] + base[1] == p[1]
                            && d[2] + base[2] == p[2]
                    })
                    .map_or(0, |r| inner_ranks[k][r] + 1)
            }
        }
    }
}

/// The order PTM receives an atom's 18 nearest neighbours in. The FCC, HCP
/// and ICO templates read the first 12, BCC the first 14, so the order decides
/// which neighbours are matched.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NeighbourOrder {
    /// By distance (the default).
    #[default]
    Distance,
    /// By decreasing Voronoi face solid angle, ties by distance, as OVITO
    /// 3.15 orders them (`ptm_preorder_neighbours` over the 18 nearest, in
    /// `PTMAlgorithm::Kernel::cacheNeighbors`). This reproduces OVITO's
    /// structure types and RMSDs atom for atom on the four reference
    /// snapshots (design §14.1), at the cost of one Voronoi cell per atom.
    Voronoi,
}

impl Ptm {
    /// PTM over a frame, one compact record per atom, neighbours by distance.
    pub fn compute(
        &self,
        cell: &crate::cell::UnitCell,
        positions: &[crate::cell::Coord3],
        par: &crate::exec::Par,
    ) -> Result<Vec<PtmAtom>> {
        self.compute_ordered(cell, positions, par, NeighbourOrder::Distance)
    }

    /// [`Ptm::compute`] with a choice of [`NeighbourOrder`].
    pub fn compute_ordered(
        &self,
        cell: &crate::cell::UnitCell,
        positions: &[crate::cell::Coord3],
        par: &crate::exec::Par,
        order: NeighbourOrder,
    ) -> Result<Vec<PtmAtom>> {
        self.scan(cell, positions, par, order, |res, _| PtmAtom {
            structure: res.kind as u8,
            rmsd: res.rmsd as f32,
            interatomic_distance: res.interatomic_distance as f32,
            orientation: res.orientation.map(|v| v as f32),
        })
    }

    /// The per-atom scan, mapping each match result through `map`.
    fn scan<T: Default + Clone + Send>(
        &self,
        cell: &crate::cell::UnitCell,
        positions: &[crate::cell::Coord3],
        par: &crate::exec::Par,
        order: NeighbourOrder,
        map: impl Fn(&matcher::MatchResult, Env<'_>) -> T + Sync + Send,
    ) -> Result<Vec<T>> {
        let n = positions.len();
        // An exact, unbounded k-NN query (design §9.2): OVITO's semantics,
        // with no density-derived radius to under-coordinate a hot lattice or
        // to blow up in vacuum.
        let index = {
            let _t = prof::StageTimer::start("ptm.knn_build");
            crate::neighbor::KnnIndex::build(cell, positions, par)?
        };
        // Each atom is independent, so the scan parallelises exactly; results
        // land in a pre-sized slot, keeping the output deterministic.
        let _t_scan = prof::StageTimer::start("ptm.scan");
        let mut results = vec![T::default(); n];
        struct Scratch {
            knn: crate::neighbor::KnnScratch,
            nbrs: Vec<(u32, [f64; 3], f64)>,
            sorted: Vec<(u32, [f64; 3], f64)>,
            inner: [Vec<(u32, [f64; 3], f64)>; 4],
            rank: Vec<u8>,
            inner_rank: [Vec<u8>; 4],
            nb: nbrlist::Neighbourhood,
            ch: hull::ConvexHull,
            res: matcher::MatchResult,
        }
        let two_shell = self.flags & check::TWO_SHELL != 0;
        let want_dmn = self.flags & (check::DCUB | check::DHEX) != 0;
        let want_grp = self.flags & check::GRAPHENE != 0;
        // One atom's neighbour list in the order PTM receives it: what the
        // host's `get_neighbours` callback returns, for the central atom and,
        // in the two-shell structures, for each inner neighbour too.
        // `rank[k]` receives the distance rank of the k-th entry (upstream's
        // correspondences, read only by the opt-in outputs).
        let gather = |i: usize,
                      knn: &mut crate::neighbor::KnnScratch,
                      out: &mut Vec<(u32, [f64; 3], f64)>,
                      tmp: &mut Vec<(u32, [f64; 3], f64)>,
                      rank: &mut Vec<u8>| {
            index.knn(i, positions[i], nbrlist::MAX_INPUT_POINTS - 1, knn, out);
            rank.clear();
            if order == NeighbourOrder::Voronoi {
                let pts: Vec<[f64; 3]> = out.iter().map(|e| e.1).collect();
                // A stable sort, as upstream's `std::stable_sort`.
                let ord = voronoi::order_neighbours(&pts);
                tmp.clear();
                tmp.extend(ord.iter().map(|o| out[o.index]));
                rank.extend(ord.iter().map(|o| o.index as u8));
                std::mem::swap(out, tmp);
            } else {
                rank.extend(0..out.len() as u8);
            }
        };
        par.try_fill_with(
            &mut results,
            || Scratch {
                knn: Default::default(),
                nbrs: Vec::with_capacity(nbrlist::MAX_INPUT_POINTS),
                sorted: Vec::with_capacity(nbrlist::MAX_INPUT_POINTS),
                inner: Default::default(),
                rank: Vec::with_capacity(nbrlist::MAX_INPUT_POINTS),
                inner_rank: Default::default(),
                nb: Default::default(),
                ch: Default::default(),
                res: Default::default(),
            },
            |s, i, out| -> Result<()> {
                gather(i, &mut s.knn, &mut s.nbrs, &mut s.sorted, &mut s.rank);
                s.nb.set_from(&s.nbrs);
                if !two_shell {
                    self.classify_gathered(&s.nb, &mut s.ch, &mut s.res);
                    *out = map(&s.res, Env::first(i, &s.nb, &s.nbrs, &s.rank));
                    return Ok(());
                }
                s.res = matcher::MatchResult::new(self.rmsd_cutoff);
                self.match_first_shell(&s.nb, &mut s.ch, &mut s.res);
                // The inner neighbours' own lists: up to 4 (diamond) or 3
                // (graphene) leading entries of the central list.
                let n_inner = if want_dmn { 4 } else { 3 }.min(s.nbrs.len());
                for k in 0..n_inner {
                    let j = s.nbrs[k].0 as usize;
                    let mut list = std::mem::take(&mut s.inner[k]);
                    let mut rank = std::mem::take(&mut s.inner_rank[k]);
                    gather(j, &mut s.knn, &mut list, &mut s.sorted, &mut rank);
                    s.inner[k] = list;
                    s.inner_rank[k] = rank;
                }
                let dmn = if want_dmn {
                    two_shell_environment_atoms(4, 3, i as u32, &s.nbrs, &s.inner[..n_inner])
                } else {
                    None
                };
                let grp = if want_grp {
                    two_shell_environment_atoms(3, 2, i as u32, &s.nbrs, &s.inner[..n_inner])
                } else {
                    None
                };
                self.match_two_shell(
                    dmn.as_ref().map(|e| &e.0),
                    grp.as_ref().map(|e| &e.0),
                    &mut s.ch,
                    &mut s.res,
                );
                self.finish(&mut s.res);
                // The environment the winning structure was matched on.
                let env = match (s.res.best_kind, &dmn, &grp) {
                    (structure::DCUB | structure::DHEX, Some(e), _)
                    | (structure::GRAPHENE, _, Some(e)) => Env {
                        points: &e.0[..],
                        atoms: EnvAtoms::Two {
                            atoms: &e.1,
                            nbrs: &s.nbrs,
                            ranks: &s.rank,
                            inner: &s.inner[..n_inner],
                            inner_ranks: &s.inner_rank[..n_inner],
                        },
                    },
                    _ => Env::first(i, &s.nb, &s.nbrs, &s.rank),
                };
                *out = map(&s.res, env);
                Ok(())
            },
        )?;
        Ok(results)
    }

    /// Frame-derived search radius covering PTM's fixed 18-neighbour
    /// environment, sized from the number density (design §14.1: a fixed
    /// radius under-coordinates a hot, expanded lattice — at 2000 K the 14th
    /// neighbour sits beyond 4 Å for up to 99.8 % of Zr atoms). `0.0` for an
    /// empty frame, where there is no density to size anything from.
    pub fn search_radius(natoms: usize, volume: f64) -> f64 {
        if natoms == 0 || volume <= 0.0 {
            return 0.0;
        }
        crate::analysis::knear_radius(nbrlist::MAX_INPUT_POINTS, natoms as f64 / volume)
    }
}

impl Analysis for Ptm {
    fn name(&self) -> &'static str {
        "ptm"
    }
    /// None: PTM builds its own k-NN index and reads no shared cell list.
    fn cutoffs(&self) -> Vec<f64> {
        Vec::new()
    }
    fn run(
        &self,
        ctx: &AnalysisContext,
        out: &mut PropertyStore,
        summary: &mut Summary,
    ) -> Result<()> {
        self.run_ordered(ctx, out, summary, NeighbourOrder::Distance)
    }
}

impl Ptm {
    /// [`Analysis::run`] with a choice of [`NeighbourOrder`] (`rmda ptm
    /// --ptm-order`); `run` is this with [`NeighbourOrder::Distance`].
    pub fn run_ordered(
        &self,
        ctx: &AnalysisContext,
        out: &mut PropertyStore,
        summary: &mut Summary,
        order: NeighbourOrder,
    ) -> Result<()> {
        self.run_extended(ctx, out, summary, order, 0)
    }

    /// [`Ptm::run_ordered`] plus the opt-in [`output`] columns selected by
    /// `outputs` (a bitmask; `0` is exactly `run_ordered`).
    ///
    /// The extra columns (`StructureOrdering`, `StructureDeformationGradient`,
    /// `StructureTemplate`, ...; see [`output`]) follow OVITO: they describe the best fit whether
    /// or not the RMSD cutoff accepted it, and are zero where nothing matched.
    /// The ordering compares the frame's particle types.
    pub fn run_extended(
        &self,
        ctx: &AnalysisContext,
        out: &mut PropertyStore,
        summary: &mut Summary,
        order: NeighbourOrder,
        outputs: u32,
    ) -> Result<()> {
        let n = ctx.frame.natoms();
        let positions = &ctx.frame.particles.positions[..n];
        let mut types = vec![structure::OTHER; n];
        let atom = |res: &matcher::MatchResult| AtomResult {
            kind: res.kind,
            rmsd: res.rmsd,
            scale: res.scale,
            interatomic_distance: res.interatomic_distance,
            lattice_constant: res.lattice_constant,
            orientation: res.orientation,
        };
        let (results, extras) = if outputs & output::ALL == 0 {
            let r = self.scan(&ctx.frame.cell, positions, ctx.par, order, |res, _| {
                atom(res)
            })?;
            (r, None)
        } else {
            let ptypes = &ctx.frame.particles.types;
            let both = self.scan(&ctx.frame.cell, positions, ctx.par, order, |res, env| {
                (atom(res), extras_of(res, &env, ptypes))
            })?;
            let (r, e): (Vec<_>, Vec<_>) = both.into_iter().unzip();
            (r, Some(e))
        };

        // Split the per-atom records into columns before registering. The
        // histogram below needs the types, so build the summary first and hand
        // the vector over rather than cloning it.
        let mut rmsds = Vec::with_capacity(n);
        let mut scales = Vec::with_capacity(n);
        let mut interatomic_distances = Vec::with_capacity(n);
        let mut lattice_constants = Vec::with_capacity(n);
        let mut orientations = Vec::with_capacity(4 * n);
        for (i, r) in results.into_iter().enumerate() {
            types[i] = r.kind;
            rmsds.push(r.rmsd);
            scales.push(r.scale);
            interatomic_distances.push(r.interatomic_distance);
            lattice_constants.push(r.lattice_constant);
            orientations.extend_from_slice(&r.orientation);
        }

        for (code, name) in [
            (structure::OTHER, "other"),
            (structure::FCC, "fcc"),
            (structure::HCP, "hcp"),
            (structure::BCC, "bcc"),
            (structure::ICO, "ico"),
            (structure::SC, "sc"),
        ] {
            let n_at = types.iter().filter(|&&t| t == code).count();
            summary.merge_from(Summary::scalar(&format!("ptm_{name}"), n_at as f64));
        }
        // The two-shell structures are reported only when they were checked,
        // so the summary of the default (OVITO) set has no entries for them.
        for (bit, code, name) in [
            (check::DCUB, structure::DCUB, "dcub"),
            (check::DHEX, structure::DHEX, "dhex"),
            (check::GRAPHENE, structure::GRAPHENE, "graphene"),
        ] {
            if self.flags & bit != 0 {
                let n_at = types.iter().filter(|&&t| t == code).count();
                summary.merge_from(Summary::scalar(&format!("ptm_{name}"), n_at as f64));
            }
        }

        out.register_scalar(crate::data::property::DType::U8, "StructureType", types);
        out.register_scalar(crate::data::property::DType::F64, "StructureRmsd", rmsds);
        out.register_scalar(crate::data::property::DType::F64, "StructureScale", scales);
        out.register_scalar(
            crate::data::property::DType::F64,
            "StructureInteratomicDistance",
            interatomic_distances,
        );
        out.register_scalar(
            crate::data::property::DType::F64,
            "StructureLatticeConstant",
            lattice_constants,
        );
        out.register_flat(
            crate::data::property::DType::F64,
            "StructureOrientation",
            4,
            orientations,
        );
        if let Some(extras) = extras {
            use crate::data::property::DType;
            if outputs & output::ORDERING != 0 {
                let ord: Vec<u8> = extras.iter().map(|e| e.ordering).collect();
                for (code, name) in [
                    (remap::ordering::NONE, "none"),
                    (remap::ordering::PURE, "pure"),
                    (remap::ordering::L10, "l10"),
                    (remap::ordering::L12_A, "l12_a"),
                    (remap::ordering::L12_B, "l12_b"),
                    (remap::ordering::B2, "b2"),
                    (remap::ordering::ZINCBLENDE_WURTZITE, "zincblende_wurtzite"),
                    (remap::ordering::BORON_NITRIDE, "boron_nitride"),
                ] {
                    let n_at = ord.iter().filter(|&&o| o == code).count();
                    summary.merge_from(Summary::scalar(
                        &format!("ptm_ordering_{name}"),
                        n_at as f64,
                    ));
                }
                out.register_scalar(DType::U8, "StructureOrdering", ord);
            }
            if outputs & output::DEFORMATION_GRADIENT != 0 {
                let f: Vec<f64> = extras.iter().flat_map(|e| e.f).collect();
                out.register_flat(DType::F64, "StructureDeformationGradient", 9, f);
            }
            if outputs & output::TEMPLATE != 0 {
                let t: Vec<u8> = extras.iter().map(|e| e.template).collect();
                out.register_scalar(DType::U8, "StructureTemplate", t);
            }
            if outputs & output::CORRESPONDENCES != 0 {
                let c: Vec<i64> = extras.iter().map(|e| e.correspondences).collect();
                out.register(crate::data::property::Property::from_data(
                    "Correspondences",
                    1,
                    crate::data::property::TypedVec::I64(c),
                ));
            }
            if outputs & output::RESIDUAL != 0 {
                let r: Vec<f64> = extras.iter().flat_map(|e| e.f_res).collect();
                out.register_flat(DType::F64, "StructureDeformationResidual", 3, r);
            }
            if outputs & output::POLAR != 0 {
                let u: Vec<f64> = extras.iter().flat_map(|e| e.u).collect();
                out.register_flat(DType::F64, "StructurePolarU", 9, u);
                let p: Vec<f64> = extras.iter().flat_map(|e| e.p).collect();
                out.register_flat(DType::F64, "StructurePolarP", 9, p);
            }
        }
        Ok(())
    }
}

/// The opt-in outputs of one atom ([`output`]).
#[derive(Clone, Copy, Debug, Default)]
struct Extras {
    ordering: u8,
    template: u8,
    f: [f64; 9],
    correspondences: i64,
    f_res: [f64; 3],
    u: [f64; 9],
    p: [f64; 9],
}

/// Upstream `output_data`'s ordering, template remap and deformation gradient
/// for the best fit (`best_kind`, regardless of the cutoff, as OVITO).
fn extras_of(res: &matcher::MatchResult, env: &Env<'_>, ptypes: &[u32]) -> Extras {
    let kind = res.best_kind;
    if kind == structure::OTHER {
        return Extras::default();
    }
    let np = refdata::RefData::for_type(kind).map_or(0, |r| r.num_nbrs + 1);
    let mut numbers = [0u32; MAX_POINTS];
    for (j, v) in numbers.iter_mut().enumerate().take(np) {
        *v = ptypes.get(env.atom(j) as usize).copied().unwrap_or(0);
    }
    let ordering = remap::find_alloy_type(kind, &res.mapping, &numbers[..np]);
    let (template, mapping) = remap::remap_template(kind, &res.q, &res.mapping);
    let f = remap::deformation_gradient(kind, template, env.points, &mapping, res.scale);
    // As upstream's `output_environment`: the correspondences in template
    // order, under the remapped mapping.
    let mut corr = [0u8; MAX_POINTS];
    for (i, c) in corr.iter_mut().enumerate().take(np) {
        *c = env.correspondence(kind, mapping[i] as usize);
    }
    let correspondences = remap::encode_correspondences(kind, &corr[..np], template);
    let f_res = remap::deformation_residual(kind, template, env.points, &mapping, res.scale, &f);
    let (u, p) = polar::polar_decomposition_3x3(&f, false);
    Extras {
        ordering,
        template,
        f,
        correspondences,
        f_res,
        u,
        p,
    }
}

/// Per-atom PTM outputs, as gathered by [`Ptm::run`] before being split into
/// property columns.
///
/// `kind` is the accepted label and may be [`structure::OTHER`] even when the
/// remaining fields are finite: the library always reports its closest
/// structure, and only the host applies `rmsd_cutoff`.
#[derive(Clone, Copy, Debug, Default)]
struct AtomResult {
    kind: i32,
    rmsd: f64,
    scale: f64,
    interatomic_distance: f64,
    lattice_constant: f64,
    /// Unit quaternion `(w, x, y, z)` in the fundamental zone of the
    /// best-fitting structure.
    orientation: [f64; 4],
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cell::UnitCell;
    use crate::neighbor::cell_list::CellList;

    /// `n^3` supercell of a primitive cell with the given basis.
    fn supercell(
        avec: [f64; 3],
        bvec: [f64; 3],
        cvec: [f64; 3],
        basis: &[[f64; 3]],
        n: usize,
    ) -> (UnitCell, Vec<[f64; 3]>) {
        let rep = |v: [f64; 3], k: usize| [v[0] * k as f64, v[1] * k as f64, v[2] * k as f64];
        let (cell, _) = UnitCell::from_general_triclinic_rot(
            rep(avec, n),
            rep(bvec, n),
            rep(cvec, n),
            [0.0; 3],
        )
        .unwrap();
        let h = cell.h_matrix();
        // `Mat3` stores cell vectors as rows, so this is `f * H`, not `H * f`.
        let to_cart = |f: [f64; 3]| {
            [
                f[0] * h.m[0][0] + f[1] * h.m[1][0] + f[2] * h.m[2][0],
                f[0] * h.m[0][1] + f[1] * h.m[1][1] + f[2] * h.m[2][1],
                f[0] * h.m[0][2] + f[1] * h.m[1][2] + f[2] * h.m[2][2],
            ]
        };
        let mut pos = Vec::new();
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    for b in basis {
                        pos.push(to_cart([
                            (i as f64 + b[0]) / n as f64,
                            (j as f64 + b[1]) / n as f64,
                            (k as f64 + b[2]) / n as f64,
                        ]));
                    }
                }
            }
        }
        (cell, pos)
    }

    /// Classify every atom of an ideal lattice with the given flags.
    fn classify_all(cell: &UnitCell, pos: &[[f64; 3]], flags: u32) -> Vec<i32> {
        let cl = CellList::build(cell, pos, 4.0).unwrap();
        let ptm = Ptm {
            rmsd_cutoff: 4.0,
            flags,
        };
        (0..pos.len())
            .map(|i| {
                let mut nb = nbrlist::Neighbourhood::default();
                let mut ch = hull::ConvexHull::default();
                let mut res = matcher::MatchResult::default();
                ptm.classify(&cl, i, &mut nb, &mut ch, &mut res).unwrap();
                res.kind
            })
            .collect()
    }

    /// The reported `lattice_constant` must be the *physical* one, i.e. the
    /// value that went into the cell, recovered through normalise -> QCP fit
    /// -> `factor / scale` -> `lattice_factor * distance`.
    ///
    /// This is the only test that would notice the whole chain being wrong
    /// while still self-consistent: a self-fit has scale ~1 and cannot
    /// distinguish a correct factor from an inverted one.
    #[test]
    fn reported_lattice_constant_recovers_the_input() {
        // (name, cell vectors, basis, flags, expected structure, lattice a)
        type Case = (
            &'static str,
            [f64; 3],
            [f64; 3],
            [f64; 3],
            &'static [[f64; 3]],
            u32,
            i32,
            f64,
        );
        let cases: &[Case] = &[
            // Simple cubic: a = NN distance, so the two must coincide.
            (
                "sc",
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                &[[0.0, 0.0, 0.0]],
                check::SC,
                structure::SC,
                1.0,
            ),
            // BCC at a = 1: NN distance is sqrt(3)/2.
            (
                "bcc",
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                &[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
                check::BCC,
                structure::BCC,
                1.0,
            ),
            // FCC at a = 1: NN distance is 1/sqrt(2).
            (
                "fcc",
                [0.0, 0.5, 0.5],
                [0.5, 0.0, 0.5],
                [0.5, 0.5, 0.0],
                &[[0.0, 0.0, 0.0]],
                check::FCC,
                structure::FCC,
                1.0,
            ),
        ];
        for (name, a, b, c, basis, flags, want, want_a) in cases {
            // Scale each case by a distinct factor so an off-by-a-constant
            // cannot pass by accident on the unit cell.
            let k = want_a * 3.7;
            let (cell, pos) = supercell(
                [a[0] * k, a[1] * k, a[2] * k],
                [b[0] * k, b[1] * k, b[2] * k],
                [c[0] * k, c[1] * k, c[2] * k],
                basis,
                4,
            );
            let cl = CellList::build(&cell, &pos, 4.0).unwrap();
            let ptm = Ptm {
                rmsd_cutoff: 4.0,
                flags: *flags,
            };
            for i in 0..pos.len() {
                let mut nb = nbrlist::Neighbourhood::default();
                let mut ch = hull::ConvexHull::default();
                let mut res = matcher::MatchResult::default();
                ptm.classify(&cl, i, &mut nb, &mut ch, &mut res).unwrap();
                assert_eq!(res.kind, *want, "{name}: atom {i} misclassified");

                // The lattice constant must be the one that went in.
                let got = res.lattice_constant;
                assert!(
                    (got - k).abs() < 1e-9 * k,
                    "{name}: atom {i} lattice constant {got}, expected {k}"
                );

                // `scale` is what ties the fit to the upstream template
                // scaling, so check it against the structure's own factors:
                //   a = c_lat * d,  d = c_int / scale
                //     => scale = c_int * c_lat / a
                // The BCC factors are the interesting case: c_int is
                // 7 - 3.5*sqrt(3) = 0.93782, *not* the physical sqrt(3)/2,
                // because the upstream template is scaled so its own first
                // point sits at that length. It cancels out of `a` but not out
                // of `scale`.
                let expect_scale = matcher::interatomic_distance_factor(*want)
                    * matcher::lattice_constant_factor(*want)
                    / k;
                assert!(
                    (res.scale - expect_scale).abs() < 1e-9 * expect_scale,
                    "{name}: atom {i} scale {}, expected {expect_scale}",
                    res.scale
                );

                // And the nearest-neighbour distance must be the physical one.
                // For BCC that is a*sqrt(3)/2; for FCC a/sqrt(2); for SC a.
                let expect_d = k / matcher::lattice_constant_factor(*want);
                assert!(
                    (res.interatomic_distance - expect_d).abs() < 1e-9 * k,
                    "{name}: atom {i} NN distance {}, expected {expect_d}",
                    res.interatomic_distance
                );
            }
        }
    }

    /// End-to-end checks on the reported orientation.
    ///
    /// Two things are asserted here. The orientation must be a genuine unit
    /// quaternion, and it must actually respond to a rotation of the lattice --
    /// together with a negative control, so a reduction that silently returned
    /// the identity would fail.
    ///
    /// What is deliberately *not* asserted is that symmetry-related lattices
    /// reduce to the same quaternion. It is tempting, and it is what grain
    /// segmentation (design §15) would want, but upstream's reduction is a
    /// single argmax fold, not a proven canonical form on the symmetry coset,
    /// and a perfect FCC lattice rotated 90 degrees about z was observed to give
    /// a representative differing from the unrotated one by a 180-degree turn
    /// about x. Whether upstream agrees could not be established, so the
    /// property is not claimed here. A consumer that needs unique
    /// representatives must compare orientations modulo the point group
    /// explicitly rather than assume the reduction did it.
    #[test]
    fn orientation_is_a_unit_quaternion_and_tracks_rotation() {
        // Unit quaternion for a rotation of `angle` about `axis`.
        let axis_angle = |axis: [f64; 3], angle: f64| {
            let n: f64 = axis.iter().map(|c| c * c).sum::<f64>().sqrt();
            let a = axis.map(|c| c / n);
            let s = (angle / 2.0).sin();
            [(angle / 2.0).cos(), s * a[0], s * a[1], s * a[2]]
        };
        // Rotate `v` by the quaternion `q` (w, x, y, z).
        let rotate = |q: [f64; 4], v: [f64; 3]| {
            let r = quat::quat_rot(
                quat::quat_rot(q, [0.0, v[0], v[1], v[2]]),
                [q[0], -q[1], -q[2], -q[3]],
            );
            [r[1], r[2], r[3]]
        };

        // FCC conventional cell, rotated by `q` throughout.
        let a = [0.0, 0.5, 0.5];
        let b = [0.5, 0.0, 0.5];
        let c = [0.5, 0.5, 0.0];
        let basis: &[[f64; 3]] = &[[0.0, 0.0, 0.0]];

        let orientation_for = |q: [f64; 4]| {
            let k = 3.7;
            let (cell, pos) = supercell(
                rotate(q, [a[0] * k, a[1] * k, a[2] * k]),
                rotate(q, [b[0] * k, b[1] * k, b[2] * k]),
                rotate(q, [c[0] * k, c[1] * k, c[2] * k]),
                basis,
                4,
            );
            let cl = CellList::build(&cell, &pos, 4.0).unwrap();
            let ptm = Ptm {
                rmsd_cutoff: 4.0,
                flags: check::FCC,
            };
            let mut nb = nbrlist::Neighbourhood::default();
            let mut ch = hull::ConvexHull::default();
            let mut res = matcher::MatchResult::default();
            ptm.classify(&cl, 0, &mut nb, &mut ch, &mut res).unwrap();
            assert_eq!(res.kind, structure::FCC, "rotated lattice misclassified");
            // Same physical lattice, so the recovered length cannot depend on
            // the orientation.
            assert!(
                (res.lattice_constant - k).abs() < 1e-9 * k,
                "lattice constant moved under rotation"
            );
            let n2: f64 = res.orientation.iter().map(|c| c * c).sum();
            assert!(
                (n2 - 1.0).abs() < 1e-12,
                "orientation is not a unit quaternion: |q|^2 = {n2}"
            );
            assert!(
                res.orientation[0] >= 0.0,
                "orientation was left outside the fundamental zone: {:?}",
                res.orientation
            );
            res.orientation
        };

        let base = orientation_for([1.0, 0.0, 0.0, 0.0]);
        let turned = orientation_for(axis_angle([0.31, -0.17, 0.93], 0.44));
        let differs = (0..4).any(|i| (turned[i] - base[i]).abs() > 1e-3);
        assert!(
            differs,
            "a genuine rotation left the orientation unchanged: {base:?}"
        );
    }

    /// The headline test: each ideal lattice must classify as itself, through
    /// the real gather -> normalise -> hull -> match path.
    #[test]
    fn ideal_lattices_classify_as_themselves() {
        // Each case carries its own flags: SC is deliberately *not* in
        // `check::DEFAULT`, so asking for it needs the bit set explicitly.
        type Case = (
            &'static str,
            [f64; 3],
            [f64; 3],
            [f64; 3],
            &'static [[f64; 3]],
            u32,
            i32,
        );
        let cases: &[Case] = &[
            (
                "sc",
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                &[[0.0, 0.0, 0.0]],
                check::SC,
                structure::SC,
            ),
            (
                "bcc",
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                &[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
                check::BCC,
                structure::BCC,
            ),
            (
                "fcc",
                [0.0, 0.5, 0.5],
                [0.5, 0.0, 0.5],
                [0.5, 0.5, 0.0],
                &[[0.0, 0.0, 0.0]],
                check::FCC,
                structure::FCC,
            ),
            (
                "hcp",
                [1.0, 0.0, 0.0],
                [-0.5, 3f64.sqrt() / 2.0, 0.0],
                [0.0, 0.0, (8.0f64 / 3.0).sqrt()],
                &[[0.0, 0.0, 0.0], [2.0 / 3.0, 1.0 / 3.0, 0.5]],
                check::HCP,
                structure::HCP,
            ),
        ];
        for (name, a, b, c, basis, flags, want) in cases {
            let (cell, pos) = supercell(*a, *b, *c, basis, 4);
            let got = classify_all(&cell, &pos, *flags);
            let bad = got.iter().filter(|&&t| t != *want).count();
            assert_eq!(bad, 0, "{name}: {bad} of {} not {want}: {got:?}", got.len());
        }
    }

    /// SC only matches when SC is actually requested.
    #[test]
    fn sc_is_only_reported_when_requested() {
        let (cell, pos) = supercell(
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            &[[0.0, 0.0, 0.0]],
            4,
        );
        let got = classify_all(
            &cell,
            &pos,
            check::FCC | check::HCP | check::ICO | check::BCC,
        );
        assert!(
            got.iter().all(|&t| t == structure::OTHER),
            "SC lattice accepted without the SC flag: {got:?}"
        );
    }

    /// Cubic diamond conventional basis (fractional).
    const DIA: [[f64; 3]; 8] = [
        [0.0, 0.0, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0],
        [0.25, 0.25, 0.25],
        [0.25, 0.75, 0.75],
        [0.75, 0.25, 0.75],
        [0.75, 0.75, 0.25],
    ];

    /// Hexagonal cell vectors `a`, `c`.
    fn hex(a: f64, c: f64) -> ([f64; 3], [f64; 3], [f64; 3]) {
        (
            [a, 0.0, 0.0],
            [-0.5 * a, a * 3f64.sqrt() / 2.0, 0.0],
            [0.0, 0.0, c],
        )
    }

    fn compute_all(cell: &UnitCell, pos: &[[f64; 3]], flags: u32) -> Vec<PtmAtom> {
        Ptm {
            rmsd_cutoff: 0.1,
            flags,
        }
        .compute(cell, pos, &crate::exec::Par::new(1))
        .unwrap()
    }

    /// The two-shell structures classify as themselves, with a self-fit RMSD
    /// and the physical lattice constant and nearest-neighbour distance, in
    /// both neighbour orders and whatever else is enabled.
    #[test]
    fn two_shell_lattices_classify_as_themselves() {
        let a_si = 5.431;
        let d = a_si * 3f64.sqrt() / 4.0;
        let (ha, hb, hc) = hex(
            a_si / 2f64.sqrt(),
            a_si / 2f64.sqrt() * (8.0f64 / 3.0).sqrt(),
        );
        let (ga, gb, gc) = hex(1.42 * 3f64.sqrt(), 20.0);
        type Case = (
            &'static str,
            [f64; 3],
            [f64; 3],
            [f64; 3],
            Vec<[f64; 3]>,
            i32,
            f64,
        );
        let cases: Vec<Case> = vec![
            (
                "dcub",
                [a_si, 0.0, 0.0],
                [0.0, a_si, 0.0],
                [0.0, 0.0, a_si],
                DIA.to_vec(),
                structure::DCUB,
                d,
            ),
            (
                "dhex",
                ha,
                hb,
                hc,
                vec![
                    [1.0 / 3.0, 2.0 / 3.0, 0.0],
                    [2.0 / 3.0, 1.0 / 3.0, 0.5],
                    [1.0 / 3.0, 2.0 / 3.0, 0.375],
                    [2.0 / 3.0, 1.0 / 3.0, 0.875],
                ],
                structure::DHEX,
                d,
            ),
            (
                "graphene",
                ga,
                gb,
                gc,
                vec![[0.0, 0.0, 0.5], [1.0 / 3.0, 2.0 / 3.0, 0.5]],
                structure::GRAPHENE,
                1.42,
            ),
        ];
        for (name, a, b, c, basis, want, nn) in cases {
            // Graphene layers sit a whole `c` (20 Å) apart: isolated sheets.
            let (cell, pos) = supercell(
                a,
                b,
                c,
                &basis,
                if want == structure::GRAPHENE { 4 } else { 3 },
            );
            let own = match want {
                structure::DCUB => check::DCUB,
                structure::DHEX => check::DHEX,
                _ => check::GRAPHENE,
            };
            for flags in [check::ALL, check::TWO_SHELL, check::OVITO_DEFAULT | own] {
                for order in [NeighbourOrder::Distance, NeighbourOrder::Voronoi] {
                    let got = Ptm {
                        rmsd_cutoff: 0.1,
                        flags,
                    }
                    .compute_ordered(&cell, &pos, &crate::exec::Par::new(1), order)
                    .unwrap();
                    for (i, r) in got.iter().enumerate() {
                        assert_eq!(
                            r.structure(),
                            want,
                            "{name}: atom {i} flags {flags:#x} {order:?}"
                        );
                        assert!(r.rmsd < 1e-6, "{name}: atom {i} rmsd {}", r.rmsd);
                        assert!(
                            (r.interatomic_distance as f64 - nn).abs() < 1e-5,
                            "{name}: atom {i} nn {} want {nn}",
                            r.interatomic_distance
                        );
                    }
                }
            }
        }
    }

    /// The two-shell structures are opt-in: the default (OVITO) set never
    /// reports them, and enabling them leaves an FCC/BCC/HCP crystal alone.
    #[test]
    fn two_shell_structures_are_opt_in() {
        let a_si = 5.431;
        let (cell, pos) = supercell(
            [a_si, 0.0, 0.0],
            [0.0, a_si, 0.0],
            [0.0, 0.0, a_si],
            &DIA,
            3,
        );
        assert!(compute_all(&cell, &pos, check::OVITO_DEFAULT)
            .iter()
            .all(|r| r.structure() == structure::OTHER));
        let (cell, pos) = supercell(
            [0.0, 1.8, 1.8],
            [1.8, 0.0, 1.8],
            [1.8, 1.8, 0.0],
            &[[0.0, 0.0, 0.0]],
            4,
        );
        let base = compute_all(&cell, &pos, check::OVITO_DEFAULT);
        let all = compute_all(&cell, &pos, check::ALL);
        for (x, y) in base.iter().zip(&all) {
            assert_eq!(x.structure(), structure::FCC);
            assert_eq!(y.structure(), structure::FCC);
            assert_eq!(x.rmsd, y.rmsd);
            assert_eq!(x.orientation, y.orientation);
        }
    }

    /// Two-shell layout: inner atoms first, then each inner atom's outer
    /// atoms in its own block, picked by rank and never twice.
    #[test]
    fn two_shell_environment_layout() {
        // Centre 0 with inner atoms 1 and 2. Atom 4 is a candidate of both at
        // the same rank; the first inner atom takes it.
        let e = |j: u32, x: f64| (j, [x, 0.0, 0.0], x * x);
        let central = vec![e(1, 1.0), e(2, -1.0), e(9, 3.0)];
        let inner = vec![
            vec![e(0, -1.0), e(3, 1.0), e(4, 2.0), e(5, 3.0)],
            vec![e(0, 1.0), e(6, -1.0), e(4, 0.5), e(7, -2.0)],
        ];
        let p = two_shell_environment(2, 2, 0, &central, &inner).unwrap();
        // inner
        assert_eq!(p[1][0], 1.0);
        assert_eq!(p[2][0], -1.0);
        // atom 1's block: rank 2 -> 3 (x=1+1), rank 3 -> 4 (x=1+2)
        assert_eq!(p[3][0], 2.0);
        assert_eq!(p[4][0], 3.0);
        // atom 2's block: rank 2 -> 6 (x=-2), rank 3 -> 4 is taken, rank 4 -> 7
        assert_eq!(p[5][0], -2.0);
        assert_eq!(p[6][0], -3.0);
        // Not enough outer candidates: None.
        assert!(two_shell_environment(2, 4, 0, &central, &inner).is_none());
    }
}
