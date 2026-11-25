//! Per-grain Wigner–Seitz defect analysis (design §16).
//!
//! For each grain (labels supplied by the caller: a region, a column, or a
//! segmentation):
//!
//! 1. the majority crystalline PTM structure, mean orientation and lattice
//!    constant seed a reference;
//! 2. [`fit`] fits a full affine lattice (rotation, strain, shear) and, on
//!    periodic axes the grain spans, makes it commensurate with the box;
//! 3. [`census`] assigns every atom to its nearest site and counts vacancies
//!    and interstitials over the occupied-site domain, split into interior
//!    and grain-boundary classes, each of which conserves atoms exactly.
//!
//! Quality gates (pair match fraction, RMS displacement, off-lattice
//! fraction) decide whether a grain's counts are reported. Every metric is
//! reported either way, so a failing gate is visible in the output.
//!
//! The global reference is a single affine lattice per grain; grains that
//! are not homogeneously strained go to the patch-local reference
//! ([`patch`], design §16.3c) or the local census (design §16.3d).

pub mod antisite;
pub mod census;
pub mod clusters;
pub mod explicit;
pub mod explicit_ws;
pub mod fit;
pub mod frame;
pub mod lattice;
mod local;
pub mod patch;
pub mod voids;

use crate::analysis::ptm::{structure, PtmAtom};
use crate::cell::{Coord3, Mat3, UnitCell};
use crate::error::Result;
use crate::exec::Par;
use crate::neighbor::{KnnIndex, KnnScratch};
pub use antisite::{Antisite, GrainAntisites};
pub use census::ClassCount;
use lattice::Motif;
pub use local::LocalStats;
use patch::NeighbourSource;

/// A total order on positions that does not depend on the order atoms are
/// stored in: the coordinates wrapped into the box, each as an f64 total
/// order.
pub(crate) fn place_key(cell: &UnitCell, x: Coord3) -> [u64; 3] {
    let w = cell.wrap(x);
    let bits = |v: f64| {
        let b = v.to_bits();
        if b >> 63 == 1 {
            !b
        } else {
            b | (1 << 63)
        }
    };
    [bits(w[0]), bits(w[1]), bits(w[2])]
}

/// About `want` of `items` (all of them when there are no more), chosen and
/// ordered by position alone: an item is kept when a hash of its position
/// falls in one of `len / want` classes, and the kept ones are sorted by
/// [`place_key`]. A fit on them, sums included, is then the same whatever
/// the order of atoms in the file; a stride over that order was not.
pub(crate) fn place_sample<T: Copy>(
    cell: &UnitCell,
    items: &[T],
    pos: impl Fn(T) -> Coord3,
    want: usize,
) -> Vec<T> {
    let stride = (items.len() / want.max(1)).max(1) as u64;
    let mut v: Vec<([u64; 3], T)> = items
        .iter()
        .filter_map(|&it| {
            let k = place_key(cell, pos(it));
            if stride > 1 {
                // splitmix64 over the three keys.
                let mut h = k[0] ^ k[1].rotate_left(21) ^ k[2].rotate_left(42);
                h = (h ^ (h >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                h = (h ^ (h >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                h ^= h >> 31;
                if h % stride != 0 {
                    return None;
                }
            }
            Some((k, it))
        })
        .collect();
    v.sort_unstable_by_key(|e| e.0);
    v.into_iter().map(|e| e.1).collect()
}

/// Settings (design §16.3 gates).
#[derive(Debug, Clone)]
pub struct DefectsConfig {
    /// Boundary zone width, Å. `None` means 1.5 first-neighbour distances.
    pub boundary_width: Option<f64>,
    /// Grains with fewer atoms are skipped.
    pub min_atoms: usize,
    /// Grains whose crystalline fraction is below this are skipped.
    pub min_crystalline_fraction: f64,
    /// Domain-filling iteration cap (cavity radius it can fill, in shells).
    pub max_domain_iterations: usize,
    /// Gate: minimum fraction of neighbour pairs the fitted lattice matches.
    pub min_pair_fraction: f64,
    /// Gate: maximum RMS displacement, as a fraction of the fitted bond.
    pub max_rms_displacement: f64,
    /// Gate: maximum fraction of atoms more than 0.35 bonds off their site.
    pub max_offlattice_fraction: f64,
    /// Gate (patch reference): minimum fraction of atoms on which a patch's
    /// site labels agree with its neighbour's. A dislocation or other lattice
    /// discontinuity inside a patch breaks the shared index space.
    pub min_phase_agreement: f64,
    /// Gate (patch reference): minimum agreement between any two
    /// face-adjacent patches, placed from each other or not. A loop or
    /// dislocation wider than a patch leaves a face whose two sides are a
    /// Burgers vector apart (agreement near 0) while every placement still
    /// agrees; ambiguity near a loop's core costs only a few percent.
    pub min_face_agreement: f64,
    /// The grain-wide fit samples at most this many crystalline atoms' first
    /// shells (patches sample their own).
    pub max_fit_atoms: usize,
    /// Which reference to count against.
    pub reference: ReferenceMode,
    /// Patch edge length, Å. `None` means 12 first-neighbour distances.
    pub patch_size: Option<f64>,
    /// Local census: how clusters that overlap accepted regions are counted.
    pub local_merge: LocalMerge,
    /// Global and patch censuses: an atom farther than 0.3 bonds from its
    /// nearest site takes the site its neighbours point to, as the local
    /// census's regions do. Off (the default), every atom goes to its
    /// nearest site: Wigner–Seitz, as OVITO counts, which puts spurious
    /// Frenkel pairs inside large loops (13-41 per a/2<111> loop in W),
    /// where the crystal is displaced by nearly b/2 from the lattice.
    pub resolve_ambiguous: bool,
}

/// How the local census counts a cluster that overlaps regions already
/// accepted (design §16.3d).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LocalMerge {
    /// Try the union with the regions it meets (up to six rounds), and count
    /// it as a partial region when the union is no region. The most accurate
    /// (thin film 999 substrate I 2005 for global's 2052); serial.
    #[default]
    Union,
    /// Count it as a partial region at once. The merge stage is ~13× cheaper
    /// and parallel, at the cost of loops larger than one cluster (substrate
    /// I 1915).
    Partial,
}

/// The reference a grain is counted against (design §16.3b, §16.3c).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReferenceMode {
    /// One affine lattice per grain.
    Global,
    /// Patch-local lattices sharing one index space.
    Patch,
    /// Global, falling back to patches when the global fit fails the
    /// displacement gates on a grain at least three patches across, and to
    /// [`ReferenceMode::Local`] when neither counts the grain.
    Auto,
    /// No lattice for the whole grain: each atom off the grain's lattice is
    /// placed by a lattice fitted to the lattice atoms around it (§16.3d).
    Local,
    /// A periodic single crystal (the grain's lattice commensurate with the
    /// box on every axis): Wigner–Seitz against that one lattice, ambiguous
    /// atoms resolved from their neighbours, past the displacement gates.
    /// Defects stored as whole atomic planes added or removed (interstitial
    /// loops grown into strips across the box, collapsed vacancy loops) are
    /// counted, which no local lattice sees, and V − I is exactly the
    /// sites the box lost or gained. A grain not commensurate with the box
    /// is counted as [`ReferenceMode::Auto`] counts it.
    Periodic,
}

impl Default for DefectsConfig {
    fn default() -> Self {
        Self {
            boundary_width: None,
            min_atoms: 100,
            min_crystalline_fraction: 0.3,
            max_domain_iterations: 12,
            min_pair_fraction: 0.8,
            max_rms_displacement: 0.15,
            max_offlattice_fraction: 0.02,
            min_phase_agreement: 0.95,
            min_face_agreement: 0.75,
            max_fit_atoms: 200_000,
            reference: ReferenceMode::Auto,
            patch_size: None,
            local_merge: LocalMerge::Union,
            resolve_ambiguous: false,
        }
    }
}

/// Why a grain has, or has no, counts.
#[derive(Debug, Clone, PartialEq)]
pub enum Status {
    Measured,
    TooSmall,
    NotCrystalline,
    /// The fit failed outright, with the reason.
    FitFailed(String),
    /// Measured, but a quality gate failed; counts are withheld.
    GateFailed(&'static str),
}

impl Status {
    pub fn label(&self) -> &str {
        match self {
            Status::Measured => "measured",
            Status::TooSmall => "too_small",
            Status::NotCrystalline => "not_crystalline",
            Status::FitFailed(_) => "fit_failed",
            Status::GateFailed(g) => g,
        }
    }
}

/// One grain's result.
#[derive(Debug, Clone)]
pub struct GrainReport {
    pub grain: i32,
    pub status: Status,
    pub n_atoms: usize,
    pub n_crystalline: usize,
    /// PTM structure code of the reference, `OTHER` when none.
    pub structure: i32,
    /// Fitted cell (columns, Cartesian) and origin.
    pub t: [[f64; 3]; 3],
    pub origin: Coord3,
    /// Fitted first-neighbour distance, Å.
    pub bond: f64,
    pub pair_fraction: f64,
    pub pair_rms: f64,
    pub snapped: [bool; 3],
    /// A grain that is the whole periodic box, whose loops round the box
    /// disagree (a box-spanning dislocation), and none of whose candidate
    /// translations gives the box as many lattice sites as atoms: sites
    /// minus atoms under the translations used. `None` otherwise.
    pub unconserved_sites: Option<i64>,
    pub rms_displacement: f64,
    pub offlattice_fraction: f64,
    pub domain_converged: bool,
    /// `"global"`, `"patch"`, `"local"`, `"periodic"` or `"explicit"`.
    pub reference: &'static str,
    /// Patches (all, and fitted rather than inherited); 1/1 for global, and
    /// meaningless for local.
    pub n_patches: usize,
    pub n_patches_fitted: usize,
    /// Lowest phase agreement between neighbouring patches (1 for global),
    /// meaningless for local.
    pub phase_agreement: f64,
    /// Lowest agreement between any two face-adjacent patches; patch only
    /// (1 otherwise).
    pub face_agreement: f64,
    /// The local census's own figures, for a grain counted locally.
    pub local: Option<LocalStats>,
    /// Counts; meaningful only when `status == Measured`.
    pub interior: ClassCount,
    pub boundary: ClassCount,
    /// Antisites, when asked for and the reference can tell sublattices
    /// (explicit, global, patch, local); `None` otherwise.
    /// Meaningful only when `status == Measured`.
    pub antisites: Option<GrainAntisites>,
}

impl GrainReport {
    fn new(grain: i32, n_atoms: usize) -> Self {
        Self {
            grain,
            status: Status::TooSmall,
            n_atoms,
            n_crystalline: 0,
            structure: structure::OTHER,
            t: [[f64::NAN; 3]; 3],
            origin: [f64::NAN; 3],
            bond: f64::NAN,
            pair_fraction: f64::NAN,
            pair_rms: f64::NAN,
            snapped: [false; 3],
            unconserved_sites: None,
            rms_displacement: f64::NAN,
            offlattice_fraction: f64::NAN,
            domain_converged: false,
            reference: "global",
            n_patches: 1,
            n_patches_fitted: 1,
            phase_agreement: 1.0,
            face_agreement: 1.0,
            local: None,
            interior: ClassCount::default(),
            antisites: None,
            boundary: ClassCount::default(),
        }
    }
}

/// A vacancy of a measured grain.
#[derive(Debug, Clone, Copy)]
pub struct Vacancy {
    pub position: Coord3,
    pub grain: i32,
    pub boundary: bool,
}

/// An interstitial atom of a measured grain.
#[derive(Debug, Clone, Copy)]
pub struct Interstitial {
    pub atom: u32,
    pub grain: i32,
    pub boundary: bool,
}

/// The whole frame's result.
#[derive(Debug, Clone, Default)]
pub struct DefectsResult {
    pub grains: Vec<GrainReport>,
    pub vacancies: Vec<Vacancy>,
    pub interstitials: Vec<Interstitial>,
    /// Antisite atoms of measured grains (empty unless asked for).
    pub antisites: Vec<Antisite>,
    /// Per atom: distance to its site, `NaN` where its grain was not measured.
    pub displacement: Vec<f32>,
}

impl DefectsResult {
    /// Totals over measured grains.
    pub fn totals(&self) -> (ClassCount, ClassCount) {
        let mut i = ClassCount::default();
        let mut b = ClassCount::default();
        for g in self.grains.iter().filter(|g| g.status == Status::Measured) {
            for (t, s) in [(&mut i, &g.interior), (&mut b, &g.boundary)] {
                t.sites += s.sites;
                t.atoms += s.atoms;
                t.vacancies += s.vacancies;
                t.interstitials += s.interstitials;
            }
        }
        (i, b)
    }

    /// Antisite totals `(interior, boundary)` over measured grains, `None`
    /// when no grain counted antisites.
    pub fn antisite_totals(&self) -> Option<(u64, u64)> {
        let mut any = false;
        let (mut i, mut b) = (0, 0);
        for g in self.grains.iter().filter(|g| g.status == Status::Measured) {
            if let Some(a) = &g.antisites {
                any = true;
                i += a.interior;
                b += a.boundary;
            }
        }
        any.then_some((i, b))
    }

    /// Antisites per `(site type, occupant type)`, `[interior, boundary]`,
    /// over measured grains.
    pub fn antisite_pairs(&self) -> std::collections::BTreeMap<(i32, i32), [u64; 2]> {
        let mut m = std::collections::BTreeMap::new();
        for g in self.grains.iter().filter(|g| g.status == Status::Measured) {
            for (k, v) in g.antisites.iter().flat_map(|a| &a.pairs) {
                let e = m.entry(*k).or_insert([0u64; 2]);
                e[0] += v[0];
                e[1] += v[1];
            }
        }
        m
    }
}

/// What the census asks about points outside the grain.
struct Env<'a> {
    index: &'a KnnIndex,
    grain_of: &'a [i32],
    grain: i32,
    claim_radius2: f64,
    /// Boundary width squared.
    width2: f64,
    /// For a grain laid out by the bond walk: its layout, so that its own
    /// atoms across a self-seam count as another grain.
    layout: Option<Layout<'a>>,
}

/// A walked grain's unwrapped positions, by grain slot.
struct Layout<'a> {
    cell: &'a UnitCell,
    slot: &'a [u32],
    local: &'a [Coord3],
    /// Axes the lattice tiles (layout wrapped there).
    wrap: [bool; 3],
    tol2: f64,
}

impl Layout<'_> {
    /// Whether this grain's atom `j`, seen at `p` (layout frame), is
    /// somewhere else in the layout: across a seam.
    fn elsewhere(&self, j: u32, p: Coord3) -> bool {
        let l = self.local[self.slot[j as usize] as usize];
        let Ok(inv_t) = self.cell.inverse_transposed() else {
            return false;
        };
        let a = self.cell.x2lamda_pre(l, &inv_t);
        let b = self.cell.x2lamda_pre(p, &inv_t);
        let mut f = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
        for (v, w) in f.iter_mut().zip(self.wrap) {
            if w {
                *v -= v.round();
            }
        }
        let o = self.cell.lamda2x([0.0; 3]);
        let x = self.cell.lamda2x(f);
        (0..3).map(|d| (x[d] - o[d]).powi(2)).sum::<f64>() > self.tol2
    }
}

impl Env<'_> {
    /// `(claimed, boundary)` at layout point `p`, ignoring atom `skip`.
    /// Another grain's atom (or one of no grain), or this grain's own atom
    /// across a self-seam, is "other": it claims `p` when nearest and within
    /// the claim radius, and puts `p` in the boundary zone within the width.
    fn zone(&self, p: Coord3, skip: Option<u32>) -> (bool, bool) {
        const K: usize = 32;
        thread_local! {
            static BUF: std::cell::RefCell<(KnnScratch, Vec<crate::neighbor::knn::Nbr>)> =
                std::cell::RefCell::new((KnnScratch::default(), Vec::new()));
        }
        BUF.with(|b| {
            let (s, out) = &mut *b.borrow_mut();
            self.index.knn_point(p, K, s, out);
            let other = |j: u32, e: Coord3| {
                self.grain_of[j as usize] != self.grain
                    || self
                        .layout
                        .as_ref()
                        .is_some_and(|l| l.elsewhere(j, [p[0] + e[0], p[1] + e[1], p[2] + e[2]]))
            };
            let mut it = out.iter().filter(|e| Some(e.0) != skip);
            let claimed = it
                .next()
                .is_some_and(|&(j, e, d2)| d2 < self.claim_radius2 && other(j, e));
            let boundary = out
                .iter()
                .filter(|e| Some(e.0) != skip)
                .any(|&(j, e, d2)| d2 <= self.width2 && other(j, e));
            (claimed, boundary)
        })
    }
}

impl census::Surroundings for Env<'_> {
    /// Claimed: the nearest atom is another grain's and close. Boundary zone:
    /// an atom of another grain (or of none) lies within the boundary width of
    /// the site itself, the same rule that flags atoms. Judging by the nearest
    /// atom's flag instead would misclass empty sites on a mismatched boundary,
    /// whose nearest atom can sit just outside the width while the site is
    /// inside it. On a grain cut open at a self-seam, its own atoms across the
    /// seam count as another grain's.
    fn probe(&self, p: Coord3) -> (bool, bool) {
        self.zone(p, None)
    }
}

/// Run the analysis over a frame. `grain_of[i]` is atom `i`'s grain, or a
/// negative value for an atom in no grain.
pub fn run(
    cell: &UnitCell,
    positions: &[Coord3],
    grain_of: &[i32],
    ptm: &[PtmAtom],
    cfg: &DefectsConfig,
    par: &Par,
) -> Result<DefectsResult> {
    run_typed(cell, positions, grain_of, ptm, None, cfg, par)
}

/// [`run`], also counting antisites (§16.1a) when `types` (one per atom) is
/// given: from the majority species of each motif basis site per grain
/// (global and patch references) or per vote region (local census).
/// Vacancies and interstitials are the same as without `types`.
pub fn run_typed(
    cell: &UnitCell,
    positions: &[Coord3],
    grain_of: &[i32],
    ptm: &[PtmAtom],
    types: Option<&[i32]>,
    cfg: &DefectsConfig,
    par: &Par,
) -> Result<DefectsResult> {
    if types.is_some_and(|t| t.len() != positions.len()) {
        return Err(crate::error::Error::invalid(
            "defects: types do not match the atoms",
        ));
    }
    let n = positions.len();
    let index = {
        let _t = crate::prof::StageTimer::start("defects.knn_build");
        KnnIndex::build(cell, positions, par)?
    };

    // Grain membership.
    let mut ids: Vec<i32> = grain_of.iter().copied().filter(|&g| g >= 0).collect();
    ids.sort_unstable();
    ids.dedup();
    let mut members: Vec<Vec<u32>> = vec![Vec::new(); ids.len()];
    // Per atom, its index in its grain's member list (a lookup the unwrap
    // makes for every bond; a binary search there cost 2× the grain stage).
    let mut slot = vec![u32::MAX; n];
    for (i, &g) in grain_of.iter().enumerate() {
        if g >= 0 {
            let k = ids.binary_search(&g).unwrap();
            slot[i] = members[k].len() as u32;
            members[k].push(i as u32);
        }
    }

    // Boundary flags need a width; no grain is fitted yet, so the default
    // comes from the frame's mean PTM first-neighbour distance.
    let width = cfg.boundary_width.unwrap_or_else(|| {
        let (s, c) = ptm
            .iter()
            .filter(|p| p.structure() != structure::OTHER && p.interatomic_distance.is_finite())
            .fold((0.0, 0usize), |(s, c), p| {
                (s + f64::from(p.interatomic_distance), c + 1)
            });
        if c > 0 {
            1.5 * s / c as f64
        } else {
            4.0
        }
    });
    let boundary = {
        let _t = crate::prof::StageTimer::start("defects.boundary");
        boundary_flags(&index, positions, grain_of, width, par)
    };

    let mut result = DefectsResult {
        displacement: vec![f32::NAN; n],
        ..Default::default()
    };
    for (k, &g) in ids.iter().enumerate() {
        let _t = crate::prof::StageTimer::start("defects.grain");
        let atoms = &members[k];
        let mut rep = GrainReport::new(g, atoms.len());
        if atoms.len() < cfg.min_atoms {
            result.grains.push(rep);
            continue;
        }
        let local_types: Option<Vec<i32>> =
            types.map(|t| atoms.iter().map(|&i| t[i as usize]).collect());
        // Computed once: `auto`'s fallback to the local census reuses it.
        let setup = match grain_setup(cell, positions, grain_of, &index, ptm, atoms, g, cfg, par) {
            Ok(s) => s,
            Err(n_crystalline) => {
                rep.n_crystalline = n_crystalline;
                rep.status = Status::NotCrystalline;
                result.grains.push(rep);
                continue;
            }
        };
        // Periodic: the global lattice, ambiguous atoms resolved, past the
        // displacement gates (the pair-fraction gate still applies); a
        // grain not commensurate with the box on every axis, or refused,
        // is counted as auto counts it.
        let periodic_cfg = DefectsConfig {
            reference: ReferenceMode::Global,
            resolve_ambiguous: true,
            max_rms_displacement: f64::INFINITY,
            max_offlattice_fraction: 1.0,
            ..cfg.clone()
        };
        let auto_cfg = DefectsConfig {
            reference: ReferenceMode::Auto,
            ..cfg.clone()
        };
        let mut cfg = cfg;
        let mut res = if cfg.reference == ReferenceMode::Periodic {
            let r = one_grain(
                cell,
                positions,
                grain_of,
                &slot,
                &boundary,
                width,
                &index,
                ptm,
                atoms,
                &setup,
                local_types.as_deref(),
                g,
                &periodic_cfg,
                par,
            );
            let usable = match &r {
                Ok((rep, _)) => {
                    rep.status == Status::Measured
                        && rep.snapped.iter().all(|&s| s)
                        && rep.unconserved_sites.is_none()
                }
                Err(_) => false,
            };
            if usable {
                r.map(|(mut rep, c)| {
                    rep.reference = "periodic";
                    (rep, c)
                })
            } else {
                let why = match &r {
                    Ok((rep, _)) if rep.status != Status::Measured => rep.status.label().to_owned(),
                    Ok((rep, _)) if rep.unconserved_sites.is_some() => format!(
                        "a dislocation spans the box and no lattice through it has as many \
                         sites as atoms ({:+} sites): its atoms are not one crystal's",
                        rep.unconserved_sites.unwrap_or(0)
                    ),
                    Ok((rep, _)) => format!(
                        "its lattice is not commensurate with the box (snapped {:?})",
                        rep.snapped
                    ),
                    Err(e) => e.to_string(),
                };
                eprintln!(
                    "defects: grain {g}: not a periodic single crystal for --reference \
                     periodic ({why}); counted as auto counts it"
                );
                cfg = &auto_cfg;
                one_grain(
                    cell,
                    positions,
                    grain_of,
                    &slot,
                    &boundary,
                    width,
                    &index,
                    ptm,
                    atoms,
                    &setup,
                    local_types.as_deref(),
                    g,
                    cfg,
                    par,
                )
            }
        } else {
            one_grain(
                cell,
                positions,
                grain_of,
                &slot,
                &boundary,
                width,
                &index,
                ptm,
                atoms,
                &setup,
                local_types.as_deref(),
                g,
                cfg,
                par,
            )
        };
        // Auto: a grain neither one lattice nor patches can count (it bends,
        // holds a dislocation network, meets its own image across a relaxed
        // seam) is counted locally instead.
        let refused = match &res {
            Ok((r, _)) => matches!(r.status, Status::FitFailed(_) | Status::GateFailed(_)),
            Err(_) => true,
        };
        if cfg.reference == ReferenceMode::Auto && refused {
            let why = match &res {
                Ok((r, _)) => r.status.label().to_owned(),
                Err(e) => e.to_string(),
            };
            eprintln!("defects: grain {g}: {why} under global/patch; counting it locally");
            let local_cfg = DefectsConfig {
                reference: ReferenceMode::Local,
                ..cfg.clone()
            };
            let alt = one_grain(
                cell,
                positions,
                grain_of,
                &slot,
                &boundary,
                width,
                &index,
                ptm,
                atoms,
                &setup,
                local_types.as_deref(),
                g,
                &local_cfg,
                par,
            );
            if alt.is_ok() {
                res = alt;
            }
        }
        // A periodic single crystal counted locally whose V − I is far from
        // zero: defect content stored as whole planes added or removed is
        // not counted there (design §16.8a).
        if let Ok((r, _)) = &res {
            let (v, i) = (
                r.interior.vacancies + r.boundary.vacancies,
                r.interior.interstitials + r.boundary.interstitials,
            );
            let net = v as i64 - i as i64;
            if r.reference == "local"
                && r.status == Status::Measured
                && atoms.len() == positions.len()
                && cell.pbc.iter().all(|&p| p)
                && net.unsigned_abs() > 50
                && net.unsigned_abs() as f64 > 0.2 * v.max(i) as f64
            {
                eprintln!(
                    "defects: grain {g}: counted locally, V - I = {net} in a periodic box: if the \
                     frame is a single crystal that kept its sites, the local census misses defects \
                     stored as whole planes (loops grown across the box); --reference periodic \
                     counts them against the crystal's one lattice"
                );
            }
        }
        match res {
            Ok((r, census)) => {
                rep = r;
                if let Some(c) = census {
                    for (local, &atom) in atoms.iter().enumerate() {
                        result.displacement[atom as usize] = c.displacement[local];
                    }
                    if let (None, Some((v, sh))) = (&c.antisites, c.antisite_order) {
                        eprintln!(
                            "defects: grain {g}: no chemical order (Cramér's V {v:.3} < {}) and \
                             no host species (majority share {sh:.2} < {}): antisites are not \
                             defined in a random alloy; not counted",
                            antisite::ORDER_MIN,
                            antisite::SHARE_MIN
                        );
                    }
                    if let Some((a, list)) = &c.antisites {
                        rep.antisites = Some(a.clone());
                        if rep.status == Status::Measured {
                            result
                                .antisites
                                .extend(list.iter().map(|&(i, t, p, b)| Antisite {
                                    atom: atoms[i as usize],
                                    site_type: t,
                                    site_position: p,
                                    grain: g,
                                    boundary: b,
                                }));
                        }
                    }
                    if rep.status == Status::Measured {
                        result
                            .vacancies
                            .extend(c.vacancies.iter().map(|&(p, b)| Vacancy {
                                position: p,
                                grain: g,
                                boundary: b,
                            }));
                        result
                            .interstitials
                            .extend(c.interstitials.iter().map(|&(i, b)| Interstitial {
                                atom: atoms[i as usize],
                                grain: g,
                                boundary: b,
                            }));
                    }
                }
            }
            Err(e) => {
                // Keep what is known: the grain's lattice structure.
                (rep.structure, rep.n_crystalline) = majority_structure(atoms, ptm);
                rep.status = Status::FitFailed(e.to_string());
            }
        }
        result.grains.push(rep);
    }
    Ok(result)
}

/// The most common lattice structure among `atoms`, and its count.
fn majority_structure(atoms: &[u32], ptm: &[PtmAtom]) -> (i32, usize) {
    let mut counts = [0usize; structure::DHCP as usize + 1];
    for &i in atoms {
        let t = ptm[i as usize].structure();
        if (0..counts.len() as i32).contains(&t) {
            counts[t as usize] += 1;
        }
    }
    let lattice_types = [
        structure::FCC,
        structure::HCP,
        structure::BCC,
        structure::SC,
        structure::DHCP,
    ];
    let st = *lattice_types
        .iter()
        .max_by_key(|&&t| counts[t as usize])
        .unwrap();
    (st, counts[st as usize])
}

/// What both references of a grain start from, computed once even when
/// `auto` falls back from global/patch to the local census: the majority
/// lattice structure, its crystalline atoms, mean orientation and bond, and
/// a sample of first-shell neighbour vectors for the fit.
struct GrainSetup {
    st: i32,
    n_crystalline: usize,
    motif: Motif,
    crystalline_local: Vec<u32>,
    q: [f64; 4],
    bond: f64,
    pairs: Vec<Coord3>,
    expected_pairs: usize,
}

/// The grain's [`GrainSetup`], or its crystalline atom count when too few of
/// its atoms are crystalline to fit a lattice.
#[allow(clippy::too_many_arguments)]
fn grain_setup(
    cell: &UnitCell,
    positions: &[Coord3],
    grain_of: &[i32],
    index: &KnnIndex,
    ptm: &[PtmAtom],
    atoms: &[u32],
    g: i32,
    cfg: &DefectsConfig,
    par: &Par,
) -> std::result::Result<GrainSetup, usize> {
    let _t = crate::prof::StageTimer::start("defects.grain.setup");
    let (st, n_crystalline) = majority_structure(atoms, ptm);
    if (n_crystalline as f64) < cfg.min_crystalline_fraction * atoms.len() as f64 {
        return Err(n_crystalline);
    }
    let motif = Motif::for_structure(st).expect("lattice structure has a motif");
    let crystalline: Vec<u32> = atoms
        .iter()
        .copied()
        .filter(|&i| ptm[i as usize].structure() == st)
        .collect();

    // Mean orientation (symmetry-aligned) and first-neighbour distance.
    let q = {
        let qs: Vec<[f64; 4]> = crystalline.iter().map(|&i| ptm[i as usize].q()).collect();
        crate::analysis::orient::mean(qs.iter(), &crate::analysis::orient::group(st).unwrap())
    };
    let bond: f64 = {
        let v: Vec<f64> = crystalline
            .iter()
            .map(|&i| f64::from(ptm[i as usize].interatomic_distance))
            .filter(|a| a.is_finite())
            .collect();
        v.iter().sum::<f64>() / v.len().max(1) as f64
    };

    // Neighbour vectors among the grain's crystalline atoms, computed on
    // demand rather than stored: a stored table is 384 B per atom.
    let z = motif.coordination();
    let source = GrainNeighbours {
        index,
        positions,
        grain_of,
        atoms,
        grain: g,
        z,
    };
    let crystalline_local: Vec<u32> = crystalline
        .iter()
        .map(|&i| {
            atoms
                .binary_search(&i)
                .expect("crystalline atom is a grain atom") as u32
        })
        .collect();
    let sample: Vec<u32> = place_sample(
        cell,
        &crystalline_local,
        |k| positions[atoms[k as usize] as usize],
        cfg.max_fit_atoms,
    );
    // In pieces of 2048 atoms: `map_chunks` hands out 65 536 items per task,
    // which would leave a 200k-atom sample on four threads. Concatenated in order,
    // so the fit's sums are the same at any thread count.
    let (pairs, expected_pairs): (Vec<Coord3>, usize) = {
        let ranges: Vec<(usize, usize)> = (0..sample.len())
            .step_by(2048)
            .map(|lo| (lo, (lo + 2048).min(sample.len())))
            .collect();
        let gather = |&(lo, hi): &(usize, usize)| {
            let mut v = Vec::with_capacity((hi - lo) * fit::fit_neighbours(z));
            let mut e = 0usize;
            for &k in &sample[lo..hi] {
                e += source.vectors(k as usize, &mut v);
            }
            (v, e)
        };
        let parts: Vec<(Vec<Coord3>, usize)> = if par.is_parallel() {
            use rayon::prelude::*;
            par.install(|| ranges.par_iter().map(gather).collect())
        } else {
            ranges.iter().map(gather).collect()
        };
        let e = parts.iter().map(|p| p.1).sum();
        (parts.into_iter().flat_map(|p| p.0).collect(), e)
    };
    Ok(GrainSetup {
        st,
        n_crystalline,
        motif,
        crystalline_local,
        q,
        bond,
        pairs,
        expected_pairs,
    })
}

#[allow(clippy::too_many_arguments)]
fn one_grain(
    cell: &UnitCell,
    positions: &[Coord3],
    grain_of: &[i32],
    slot: &[u32],
    boundary: &[bool],
    width: f64,
    index: &KnnIndex,
    ptm: &[PtmAtom],
    atoms: &[u32],
    setup: &GrainSetup,
    types: Option<&[i32]>,
    g: i32,
    cfg: &DefectsConfig,
    par: &Par,
) -> Result<(GrainReport, Option<census::Census>)> {
    let mut rep = GrainReport::new(g, atoms.len());

    rep.n_crystalline = setup.n_crystalline;
    rep.structure = setup.st;
    let GrainSetup {
        st,
        motif,
        crystalline_local,
        q,
        bond,
        pairs,
        expected_pairs,
        ..
    } = setup;
    let (st, q, bond, expected_pairs) = (*st, *q, *bond, *expected_pairs);
    let motif = motif.clone();
    let z = motif.coordination();
    let source = GrainNeighbours {
        index,
        positions,
        grain_of,
        atoms,
        grain: g,
        z,
    };
    if cfg.reference == ReferenceMode::Local {
        let lt = crate::prof::StageTimer::start("defects.grain.local.prefit");
        let wrapped: Vec<Coord3> = atoms.iter().map(|&i| positions[i as usize]).collect();
        let pre = fit::fit(
            &fit::FitInput {
                motif: &motif,
                positions: &wrapped,
                pairs,
                expected_pairs,
                orientation: q,
                bond,
                translations: [None; 3],
            },
            cell,
            [false; 3],
            par,
        )?;
        drop(lt);
        rep.t = pre.lattice.t.m;
        rep.origin = pre.lattice.origin;
        rep.bond = pre.lattice.nn();
        rep.pair_fraction = pre.pair_fraction;
        rep.pair_rms = pre.pair_rms;
        let env = Env {
            index,
            grain_of,
            grain: g,
            claim_radius2: (0.6 * rep.bond).powi(2),
            width2: width * width,
            layout: None,
        };
        let lattice_atom = |j: usize| ptm[j].structure() == st;
        let (c, s) = {
            let _t = crate::prof::StageTimer::start("defects.grain.local");
            local::census(
                &local::LocalInput {
                    cell,
                    positions,
                    index,
                    grain_of,
                    boundary,
                    lattice_atom: &lattice_atom,
                    atoms,
                    grain: g,
                    motif: &motif,
                    t: pre.lattice.t,
                    env: &env,
                    max_domain_iterations: cfg.max_domain_iterations,
                    types,
                    merge: cfg.local_merge,
                },
                par,
            )?
        };
        eprintln!(
            "defects: grain {g}: local census over {} atoms off the lattice ({} unresolved); \
             {} clusters counted as regions ({} atoms), {} defect atoms in extended defects \
             counted atom by atom; origin-fit rms {:.3} A",
            s.defect_atoms, s.unresolved, s.regions, s.region_atoms, s.extended, s.frame_rms
        );
        rep.reference = "local";
        rep.local = Some(s);
        rep.rms_displacement = c.rms_displacement;
        rep.offlattice_fraction = c.offlattice_fraction;
        rep.domain_converged = c.domain_converged;
        rep.interior = c.interior;
        rep.boundary = c.boundary;
        rep.status = if rep.pair_fraction < cfg.min_pair_fraction {
            Status::GateFailed("gate_pair_fraction")
        } else {
            Status::Measured
        };
        return Ok((rep, Some(c)));
    }

    // Unwrap the grain on periodic axes it does not span; snap on those it
    // does. An axis the grain's projection leaves a gap on is not spanned,
    // and the grain is cut in that gap. An axis it covers is spanned when the
    // box vector is a lattice translation of a first, unsnapped fit (on the
    // wrapped positions; the neighbour vectors do not depend on wrapping).
    // Otherwise (a large non-periodic grain, or a lattice that wraps the box
    // with an offset) the grain is laid out by a walk over lattice-coherent
    // bonds, which costs a k-NN query per atom and so runs only then.
    let (covered, cut) = projection(cell, positions, atoms, bond);
    let mut walk = None;
    if covered.iter().any(|&c| c) {
        let _t = crate::prof::StageTimer::start("defects.grain.prefit");
        let wrapped: Vec<Coord3> = atoms.iter().map(|&i| positions[i as usize]).collect();
        let pre = fit::fit(
            &fit::FitInput {
                motif: &motif,
                positions: &wrapped,
                pairs,
                expected_pairs,
                orientation: q,
                bond,
                translations: [None; 3],
            },
            cell,
            covered,
            par,
        )?;
        // Walk when the box does not close on a lattice translation, or does
        // only within the snap tolerance but measurably (> 0.1 bonds): a
        // self-seam with a small offset snaps, and its strain then shows as
        // false defects along the seam. The walk keeps the snap when the
        // lattice still closes through the box.
        // An offset is told from thermal bond noise when it exceeds 3.5
        // times the per-component RMS of the fitted bond residuals.
        let nn = pre.lattice.nn();
        let sigma = pre.pair_rms / 3f64.sqrt();
        let floor = (0.1 * nn).max(3.5 * sigma);
        // A walked bond must be coherent to within thermal noise, not merely
        // nearer the lattice than a closure offset: atomsk self-seams hold
        // atoms 0.4-0.5 A off the lattice in directions no offset predicts,
        // and a 0.35-bond tolerance lets the walk cross the seam through them
        // (a 500 A box fails one grain in four; the 1270 A one all three).
        let coherent = (0.1 * nn).max(5.0 * sigma);
        let open: Vec<Coord3> = (0..3)
            .filter(|&d| covered[d] && (!pre.snapped[d] || pre.snap_residual[d] > floor))
            .map(|d| pre.snap_offset[d])
            .collect();
        if !open.is_empty() {
            walk = Some((pre.lattice, open, floor, coherent));
        }
    }
    // Whether the walk cut the grain open at a self-seam: only then do its
    // own atoms across the seam count as another grain's.
    let mut cut_open = false;
    // Lattice translations of the box vectors the walk closed loops along.
    let mut translations = [None; 3];
    let mut unconserved_sites = None;
    let (local, spans) = match walk {
        None => (place(cell, positions, atoms, covered, cut), covered),
        Some((lat, open, floor, coherent)) => {
            let _t = crate::prof::StageTimer::start("defects.grain.walk");
            let shell: Vec<Coord3> = motif
                .shell_vectors
                .iter()
                .map(|&f| lat.t.mul_v(f))
                .collect();
            let bonds = Bonds::new(
                shell,
                motif.shell_vectors.clone(),
                &open,
                &lat,
                floor,
                coherent,
            );
            let mut lattice_atom = vec![false; atoms.len()];
            for &k in crystalline_local.iter() {
                lattice_atom[k as usize] = true;
            }
            let (local, spans, seam, cands) = unwrap_grain(
                cell,
                positions,
                index,
                grain_of,
                slot,
                atoms,
                &lattice_atom,
                &bonds,
                par,
            );
            let axes: String = (0..3)
                .filter(|&d| covered[d] && !spans[d])
                .map(|d| ["x", "y", "z"][d])
                .collect();
            // A covered axis the walk did not close is a seam the grain meets
            // its own image across: a sharp one (seam atoms) or, after MD, a
            // relaxed, disordered one the walk over lattice atoms stops at.
            cut_open = !axes.is_empty();
            if cut_open {
                eprintln!(
                    "defects: grain {g} meets its own periodic image; cut open along {axes} \
                     ({seam} seam atoms)"
                );
            }
            let split: String = (0..3)
                .filter(|&d| cands[d].len() > 1 && spans[d])
                .map(|d| ["x", "y", "z"][d])
                .collect();
            translations = [0, 1, 2].map(|d| cands[d].first().map(|v| v.0));
            if !split.is_empty() {
                // The whole periodic box one grain: the translations that
                // conserve the atoms, not the majority's (see
                // [`conserving_translations`]).
                let whole = atoms.len() == positions.len()
                    && cell.pbc.iter().all(|&p| p)
                    && spans.iter().all(|&s| s);
                let pick = whole
                    .then(|| conserving_translations(&cands, motif.basis.len(), atoms.len()))
                    .flatten();
                if let Some((_, diff)) = pick.filter(|p| p.1 != 0) {
                    unconserved_sites = Some(diff);
                }
                let how = match pick {
                    Some((p, 0)) if p.iter().zip(&translations).any(|(a, b)| Some(*a) != *b) => {
                        translations = p.map(Some);
                        "the box's atoms fill the lattice of a minority's translation; \
                         snapping to it"
                    }
                    Some((p, d)) if d != 0 => {
                        translations = p.map(Some);
                        "no translation gives the box as many lattice sites as atoms; \
                         snapping to the nearest"
                    }
                    _ => "snapping to the translation most give",
                };
                eprintln!(
                    "defects: grain {g}: loops round the box along {split} disagree by a \
                     lattice vector (a dislocation); {how}"
                );
            }
            (local, spans)
        }
    };

    let fit_timer = crate::prof::StageTimer::start("defects.grain.fit");
    let fitted = fit::fit(
        &fit::FitInput {
            motif: &motif,
            positions: &local,
            pairs,
            expected_pairs,
            orientation: q,
            bond,
            translations,
        },
        cell,
        spans,
        par,
    )?;
    drop(fit_timer);
    if let Some(d) = (0..3).find(|&d| spans[d] && !fitted.snapped[d]) {
        // The lattice continues across the periodic boundary, but going
        // round the box does not close on a lattice translation: the grain
        // meets its own image across a translation fault, or holds a
        // dislocation. One lattice cannot count it.
        return Err(crate::error::Error::invalid(format!(
            "the lattice continues across the periodic boundary along {} but closes \
             {:.2} A off a lattice translation (self-contact across a fault, or a \
             dislocation)",
            ["x", "y", "z"][d],
            fitted.snap_residual[d]
        )));
    }
    let lat = &fitted.lattice;
    rep.t = lat.t.m;
    rep.origin = lat.origin;
    rep.bond = lat.nn();
    rep.pair_fraction = fitted.pair_fraction;
    rep.pair_rms = fitted.pair_rms;
    rep.snapped = fitted.snapped;
    rep.unconserved_sites = unconserved_sites;

    let env = Env {
        index,
        grain_of,
        grain: g,
        claim_radius2: (0.6 * rep.bond).powi(2),
        width2: width * width,
        layout: cut_open.then(|| Layout {
            cell,
            slot,
            local: &local,
            wrap: fitted.snapped,
            tol2: (0.5 * rep.bond).powi(2),
        }),
    };
    // A grain cut open at a self-seam: its atoms within the width of the
    // seam (of a same-grain atom that is not where the layout puts it) are
    // boundary atoms too.
    let mut local_boundary: Vec<bool> = atoms.iter().map(|&i| boundary[i as usize]).collect();
    if cut_open {
        par.fill(&mut local_boundary, |k, o| {
            if !*o {
                *o = env.zone(local[k], Some(atoms[k])).1;
            }
        });
    }
    // With `resolve_ambiguous`: a grain atom's same-grain neighbours within
    // 1.25 bonds, for its site's vote.
    let reach2 = (1.25 * rep.bond).powi(2);
    let neighbours = |k: usize, out: &mut Vec<(usize, Coord3)>| {
        thread_local! {
            static BUF: std::cell::RefCell<(crate::neighbor::KnnScratch, Vec<crate::neighbor::knn::Nbr>)> =
                std::cell::RefCell::new(Default::default());
        }
        out.clear();
        let i = atoms[k] as usize;
        BUF.with(|b| {
            let (sc, nb) = &mut *b.borrow_mut();
            index.knn(i, positions[i], 14, sc, nb);
            for e in nb.iter() {
                let j = e.0 as usize;
                if e.2 < reach2 && grain_of[j] == g && slot[j] != u32::MAX {
                    out.push((slot[j] as usize, e.1));
                }
            }
        });
    };
    let neighbours: Option<&census::Neighbours> = cfg.resolve_ambiguous.then_some(&neighbours);
    let run_census = |r: &dyn lattice::Reference| {
        census::census(
            &census::CensusInput {
                lattice: r,
                cell,
                wrap: fitted.snapped,
                positions: &local,
                boundary: &local_boundary,
                max_domain_iterations: cfg.max_domain_iterations,
                types,
                neighbours,
            },
            &env,
            par,
        )
    };
    let gate = |rep: &GrainReport| -> Status {
        if rep.pair_fraction < cfg.min_pair_fraction {
            Status::GateFailed("gate_pair_fraction")
        } else if rep.rms_displacement > cfg.max_rms_displacement * rep.bond {
            Status::GateFailed("gate_rms_displacement")
        } else if rep.offlattice_fraction > cfg.max_offlattice_fraction {
            Status::GateFailed("gate_offlattice")
        } else if rep.phase_agreement < cfg.min_phase_agreement {
            Status::GateFailed("gate_phase_agreement")
        } else if rep.face_agreement < cfg.min_face_agreement {
            Status::GateFailed("gate_face_agreement")
        } else {
            Status::Measured
        }
    };
    let record = |rep: &mut GrainReport, c: &census::Census| {
        rep.rms_displacement = c.rms_displacement;
        rep.offlattice_fraction = c.offlattice_fraction;
        rep.domain_converged = c.domain_converged;
        rep.interior = c.interior;
        rep.boundary = c.boundary;
    };

    let patch_size = cfg.patch_size.unwrap_or(12.0 * rep.bond);
    let extent = {
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for p in &local {
            for a in 0..3 {
                lo[a] = lo[a].min(p[a]);
                hi[a] = hi[a].max(p[a]);
            }
        }
        (0..3).map(|a| hi[a] - lo[a]).fold(0.0, f64::max)
    };
    let mut c = None;
    // Auto: a grain whose atoms are clearly too far from the global lattice
    // fails the displacement gates whatever the census finds, and falls back
    // to patches or the local census. A sample of its atoms says so for a
    // fraction of the census's cost (thin film as one grain: 8.6 s; the 128M
    // frame: 69 s for two grains). Only a clear failure skips the census, so
    // no grain that could pass is refused.
    let mut skip_global = false;
    if cfg.reference == ReferenceMode::Auto {
        let _t = crate::prof::StageTimer::start("defects.grain.estimate");
        let (rms, off) = displacement_estimate(cell, lat, &local, rep.bond, par);
        if rms > ESTIMATE_RMS_MARGIN * cfg.max_rms_displacement * rep.bond
            || off > ESTIMATE_OFF_MARGIN * cfg.max_offlattice_fraction
        {
            rep.rms_displacement = rms;
            rep.offlattice_fraction = off;
            rep.status = gate(&rep);
            skip_global = true;
        }
    }
    if cfg.reference != ReferenceMode::Patch && !skip_global {
        let g_c = {
            let _t = crate::prof::StageTimer::start("defects.grain.census");
            run_census(lat)?
        };
        record(&mut rep, &g_c);
        rep.status = gate(&rep);
        c = Some(g_c);
    }
    let displacement_gate = matches!(
        rep.status,
        Status::GateFailed("gate_rms_displacement") | Status::GateFailed("gate_offlattice")
    );
    let want_patch = match cfg.reference {
        ReferenceMode::Patch => true,
        ReferenceMode::Global | ReferenceMode::Local | ReferenceMode::Periodic => false,
        ReferenceMode::Auto => displacement_gate && extent >= 3.0 * patch_size,
    };
    if want_patch {
        let _t = crate::prof::StageTimer::start("defects.patch");
        let pl = patch::build(
            lat,
            fitted.snap_translation,
            cell,
            &local,
            crystalline_local,
            &source,
            patch_size,
            par,
        );
        rep.reference = "patch";
        rep.n_patches = pl.n_patches();
        rep.n_patches_fitted = pl.n_fitted();
        rep.phase_agreement = pl.min_phase_agreement;
        rep.face_agreement = pl.min_face_agreement;
        // Auto: patches whose labels already disagree are refused whatever
        // their census finds; skip it and fall back to the local census.
        let phase_fails = pl.min_phase_agreement < cfg.min_phase_agreement;
        let face_fails = pl.min_face_agreement < cfg.min_face_agreement;
        if cfg.reference == ReferenceMode::Auto && (phase_fails || face_fails) {
            rep.status = Status::GateFailed(if phase_fails {
                "gate_phase_agreement"
            } else {
                "gate_face_agreement"
            });
            return Ok((rep, None));
        }
        let p_c = run_census(&pl)?;
        record(&mut rep, &p_c);
        rep.status = gate(&rep);
        c = Some(p_c);
    }
    Ok((rep, c))
}

/// Auto skips the global census when a sample's rms displacement exceeds
/// the gate by this factor, or its off-lattice fraction by the next.
const ESTIMATE_RMS_MARGIN: f64 = 1.5;
const ESTIMATE_OFF_MARGIN: f64 = 2.0;

/// The census's displacement metrics, estimated from up to 200 000 of the
/// grain's atoms chosen by position: rms distance to the nearest site of
/// `lat`, and the fraction farther than 0.35 bonds (as the census measures
/// them). Summed in fixed chunks, so the same at any thread count.
fn displacement_estimate(
    cell: &UnitCell,
    lat: &lattice::AffineLattice,
    local: &[Coord3],
    bond: f64,
    par: &Par,
) -> (f64, f64) {
    let idx: Vec<u32> = (0..local.len() as u32).collect();
    let sample = place_sample(cell, &idx, |k| local[k as usize], 200_000);
    if sample.is_empty() {
        return (0.0, 0.0);
    }
    let off2 = (0.35 * bond).powi(2);
    let parts = par.map_chunks(sample.len(), |lo, hi| {
        let (mut s2, mut off) = (0.0f64, 0usize);
        for &k in &sample[lo..hi] {
            let d2 = lattice::Reference::nearest(lat, local[k as usize]).2;
            s2 += d2;
            off += usize::from(d2 > off2);
        }
        (s2, off)
    });
    let (s2, off) = parts
        .into_iter()
        .fold((0.0, 0usize), |a, b| (a.0 + b.0, a.1 + b.1));
    let n = sample.len() as f64;
    ((s2 / n).sqrt(), off as f64 / n)
}

/// Same-grain first-shell neighbour vectors of a grain's atoms, on demand.
struct GrainNeighbours<'a> {
    index: &'a KnnIndex,
    positions: &'a [Coord3],
    grain_of: &'a [i32],
    atoms: &'a [u32],
    grain: i32,
    z: usize,
}

impl patch::NeighbourSource for GrainNeighbours<'_> {
    fn vectors(&self, local: usize, out: &mut Vec<Coord3>) -> usize {
        thread_local! {
            static BUF: std::cell::RefCell<(KnnScratch, Vec<crate::neighbor::knn::Nbr>)> =
                std::cell::RefCell::new((KnnScratch::default(), Vec::new()));
        }
        let i = self.atoms[local] as usize;
        BUF.with(|b| {
            let (s, nb) = &mut *b.borrow_mut();
            self.index
                .knn(i, self.positions[i], fit::fit_neighbours(self.z), s, nb);
            let same = |e: &&crate::neighbor::knn::Nbr| self.grain_of[e.0 as usize] == self.grain;
            out.extend(nb.iter().filter(same).map(|e| e.1));
            // Expected first-shell pairs: the same-grain atoms among the `z`
            // nearest (`nb` is closest first). An atom of a thin layer (a
            // stacking fault's two HCP planes) has fewer than `z` there, and
            // its second shell must not stand in for the missing ones.
            nb.iter().take(self.z).filter(same).count()
        })
    }
}

/// Atoms within `width` of an atom of another grain (or of no grain).
fn boundary_flags(
    index: &KnnIndex,
    positions: &[Coord3],
    grain_of: &[i32],
    width: f64,
    par: &Par,
) -> Vec<bool> {
    const K: usize = 32;
    let w2 = width * width;
    let mut out = vec![false; positions.len()];
    par.fill(&mut out, |i, o| {
        if grain_of[i] < 0 {
            *o = true;
            return;
        }
        thread_local! {
            static BUF: std::cell::RefCell<(KnnScratch, Vec<crate::neighbor::knn::Nbr>)> =
                std::cell::RefCell::new((KnnScratch::default(), Vec::new()));
        }
        BUF.with(|b| {
            let (s, nb) = &mut *b.borrow_mut();
            index.knn(i, positions[i], K, s, nb);
            *o = nb
                .iter()
                .any(|&(j, _, d2)| d2 <= w2 && grain_of[j as usize] != grain_of[i]);
        });
    });
    out
}

/// Per periodic axis: whether the grain's projection covers it (no gap of
/// two bonds between its atoms' fractional coordinates, across the wrap
/// included), and otherwise the fraction in the middle of the largest gap,
/// where the grain can be cut without cutting a bond.
fn projection(
    cell: &UnitCell,
    positions: &[Coord3],
    atoms: &[u32],
    bond: f64,
) -> ([bool; 3], [f64; 3]) {
    let mut covered = [false; 3];
    let mut cut = [0.0f64; 3];
    let Ok(inv_t) = cell.inverse_transposed() else {
        return (covered, cut);
    };
    let heights = cell.heights();
    for d in 0..3 {
        if !cell.pbc[d] || atoms.is_empty() {
            continue;
        }
        let mut f: Vec<f64> = atoms
            .iter()
            .map(|&i| {
                let l = cell.x2lamda_pre(positions[i as usize], &inv_t)[d];
                l - l.floor()
            })
            .collect();
        f.sort_unstable_by(f64::total_cmp);
        let (mut gap, mut at) = (f[0] + 1.0 - f[f.len() - 1], f[f.len() - 1]);
        for w in f.windows(2) {
            if w[1] - w[0] > gap {
                gap = w[1] - w[0];
                at = w[0];
            }
        }
        if gap * heights[d] < 2.0 * bond {
            covered[d] = true;
        } else {
            cut[d] = at + 0.5 * gap;
        }
    }
    (covered, cut)
}

/// Positions with every atom in `[cut, cut + 1)` along each periodic axis
/// not `spanned`, and in the box along the spanned ones.
fn place(
    cell: &UnitCell,
    positions: &[Coord3],
    atoms: &[u32],
    spanned: [bool; 3],
    cut: [f64; 3],
) -> Vec<Coord3> {
    let Ok(inv_t) = cell.inverse_transposed() else {
        return atoms.iter().map(|&i| positions[i as usize]).collect();
    };
    atoms
        .iter()
        .map(|&i| {
            let mut l = cell.x2lamda_pre(positions[i as usize], &inv_t);
            for d in 0..3 {
                if cell.pbc[d] && !spanned[d] {
                    let f = l[d] - l[d].floor();
                    let c = cut[d] - cut[d].floor();
                    l[d] = l[d].floor() + f + if f < c { 1.0 } else { 0.0 };
                }
            }
            cell.lamda2x(l)
        })
        .collect()
}

/// One piece of a walk level: reached atoms (atom, parent, position and
/// lattice coordinate from that parent), the axes a loop closed on, and the
/// loops' (box shift, lattice translation) pairs.
type Scan = (
    Vec<(u32, u32, Coord3, Coord3)>,
    [bool; 3],
    Vec<([i64; 3], Coord3)>,
);

/// How the walk of [`unwrap_grain`] reads a bond: against the grain's
/// first-shell vectors and the closure offsets of the axes its lattice does
/// not tile.
pub(crate) struct Bonds {
    /// First-shell vectors of the fitted lattice, Cartesian.
    pub shell: Vec<Coord3>,
    /// The same, as lattice translations (fractional, motif cell).
    pub shell_frac: Vec<Coord3>,
    /// Closure offsets: box vector minus nearest lattice translation, for
    /// each axis the lattice does not tile, with signs and pairwise sums.
    pub offsets: Vec<Coord3>,
    /// First-neighbour distance.
    pub nn: f64,
    /// Largest deviation from a shell vector a walked bond may have, Å.
    pub coherent: f64,
}

impl Bonds {
    /// Offsets (with signs and pairwise sums) are reduced to their shortest
    /// form modulo the lattice's translations; those no longer than `floor`
    /// are dropped, since their bisector would cut into thermal bond noise.
    /// `coherent` bounds the deviation of a bond the walk may cross.
    pub fn new(
        shell: Vec<Coord3>,
        shell_frac: Vec<Coord3>,
        axis_offsets: &[Coord3],
        lat: &lattice::AffineLattice,
        floor: f64,
        coherent: f64,
    ) -> Self {
        let nn = lat.nn();
        let mut offsets: Vec<Coord3> = Vec::new();
        for (a, ra) in axis_offsets.iter().enumerate() {
            for sa in [1.0, -1.0] {
                offsets.push(ra.map(|v| sa * v));
                for rb in &axis_offsets[a + 1..] {
                    for sb in [1.0, -1.0] {
                        offsets.push([
                            sa * ra[0] + sb * rb[0],
                            sa * ra[1] + sb * rb[1],
                            sa * ra[2] + sb * rb[2],
                        ]);
                    }
                }
            }
        }
        // Shortest equivalent: subtract the nearest pure translation.
        let reduce = |o: Coord3| -> Coord3 {
            let Some(ti) = lat.t.inverse() else {
                return o;
            };
            let f = ti.mul_v(o);
            lat.motif
                .translations
                .iter()
                .map(|tr| {
                    let n: Coord3 = std::array::from_fn(|k| (f[k] - tr[k]).round() + tr[k]);
                    let p = lat.t.mul_v(n);
                    [o[0] - p[0], o[1] - p[1], o[2] - p[2]]
                })
                .min_by(|a, b| {
                    let l = |v: &Coord3| v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
                    l(a).total_cmp(&l(b))
                })
                .unwrap_or(o)
        };
        let mut offsets: Vec<Coord3> = offsets.into_iter().map(reduce).collect();
        offsets.retain(|o| o[0] * o[0] + o[1] * o[1] + o[2] * o[2] > floor * floor);
        Self {
            shell,
            shell_frac,
            offsets,
            nn,
            coherent,
        }
    }

    /// `(coherent, seam)` for bond `v`, from its deviation `dev` from the
    /// nearest first-shell vector. Seam-like: `dev` is nearer one of the
    /// closure offsets than zero (across their perpendicular bisector), the
    /// signature of a bond across a self-seam, where the grain's own image
    /// sits that offset off its lattice, and within 0.35 bonds of it (a
    /// second-shell bond is neither). Coherent: not seam-like, and within the
    /// thermal tolerance `coherent` of the shell vector. A bond off the lattice
    /// by more than noise but toward no closure offset (an atom left out of
    /// place on a seam) is neither, so the walk does not cross it either.
    #[cfg(test)]
    fn read(&self, v: Coord3) -> (bool, bool) {
        let (c, s, _) = self.read_step(v);
        (c, s)
    }

    /// [`Bonds::read`], plus the index of the nearest shell vector: the
    /// lattice step a coherent bond takes.
    fn read_step(&self, v: Coord3) -> (bool, bool, usize) {
        let d2 = |a: Coord3, b: Coord3| (0..3).map(|k| (a[k] - b[k]).powi(2)).sum::<f64>();
        let Some((wi, w)) = self
            .shell
            .iter()
            .enumerate()
            .min_by(|a, b| d2(v, *a.1).total_cmp(&d2(v, *b.1)))
        else {
            return (false, false, 0);
        };
        let dev = [v[0] - w[0], v[1] - w[1], v[2] - w[2]];
        let dev2 = d2(dev, [0.0; 3]);
        let near = (0.35 * self.nn).powi(2);
        let seam = self
            .offsets
            .iter()
            .any(|o| d2(dev, *o) < dev2 && d2(dev, *o) < near);
        (!seam && dev2 < self.coherent.powi(2), seam, wi)
    }
}

/// What [`unwrap_grain`] returns: contiguous positions, the axes a loop
/// closed on, the number of seam atoms and each axis's candidate loop
/// translations ([`loop_candidates`]).
type Unwrapped = (Vec<Coord3>, [bool; 3], usize, [Vec<(Coord3, usize)>; 3]);

/// Grain positions made contiguous by a walk over bonds, for a grain whose
/// projection covers a periodic axis that its lattice does not tile.
///
/// A breadth-first walk over same-grain bonds that are first-shell vectors
/// of the grain's lattice (among the 14 nearest neighbours) lays each
/// connected piece of the grain out without wrapping. A bond that closes a
/// loop through a periodic image means the lattice continues across the
/// periodic boundary: the grain spans that axis (returned flag), its atoms
/// wrapped back into the box along it.
///
/// **Self-seams.** A grain that meets its own periodic image (a few grains
/// in a periodic box, as atomsk builds them) does so across a translation
/// boundary whose offset is the closure offset of the box: its bonds deviate
/// from the lattice by about that offset ([`Bonds::read`]). Atoms with two or
/// more such bonds are seam atoms. The walk neither enters them nor crosses a
/// seam bond, so the grain is cut open exactly at the seam; seam atoms are
/// then attached to the side they bond to coherently. Returns the number of
/// seam atoms as well.
///
/// **Relaxed seams.** After MD a self-seam is a disordered interface whose
/// bonds no longer carry the offset. The walk therefore runs over lattice
/// atoms only (`lattice_atom`: PTM matched the grain's structure); an
/// interface of non-lattice atoms stops it, while a dislocation core (a line)
/// or a vacancy shell (closed) cannot cut the grain in two. Non-lattice atoms
/// are attached afterwards, like seam atoms. A lattice that still closes a loop through the box,
/// with no such interface, holds a fault or dislocation (the caller refuses
/// it).
///
/// Pieces of a grain not bonded to its largest piece are moved, by whole box
/// vectors, to the image nearest it.
#[allow(clippy::too_many_arguments)]
fn unwrap_grain(
    cell: &UnitCell,
    positions: &[Coord3],
    index: &KnnIndex,
    grain_of: &[i32],
    slot: &[u32],
    atoms: &[u32],
    lattice_atom: &[bool],
    bonds: &Bonds,
    par: &Par,
) -> Unwrapped {
    let n = atoms.len();
    let wrapped: Vec<Coord3> = atoms.iter().map(|&i| positions[i as usize]).collect();
    let Ok(inv_t) = cell.inverse_transposed() else {
        return (wrapped, [false; 3], 0, Default::default());
    };
    let frac = |v: Coord3| -> Coord3 {
        let a = cell.x2lamda_pre(cell.origin, &inv_t);
        let b = cell.x2lamda_pre(
            [
                cell.origin[0] + v[0],
                cell.origin[1] + v[1],
                cell.origin[2] + v[2],
            ],
            &inv_t,
        );
        [b[0] - a[0], b[1] - a[1], b[2] - a[2]]
    };
    let g = grain_of[atoms[0] as usize];
    thread_local! {
        static BUF: std::cell::RefCell<(KnnScratch, Vec<crate::neighbor::knn::Nbr>)> =
            std::cell::RefCell::new((KnnScratch::default(), Vec::new()));
    }
    // Same-grain neighbours of grain atom k: (grain index, vector, coherent,
    // seam-like, lattice step).
    let neighbours = |k: usize, f: &mut dyn FnMut(usize, Coord3, bool, bool, Coord3)| {
        BUF.with(|b| {
            let (s, nb) = &mut *b.borrow_mut();
            let i = atoms[k] as usize;
            index.knn(i, positions[i], 14, s, nb);
            for &(j, v, _) in nb.iter() {
                if grain_of[j as usize] == g {
                    let (c, sm, w) = bonds.read_step(v);
                    let step = bonds.shell_frac.get(w).copied().unwrap_or([0.0; 3]);
                    f(slot[j as usize] as usize, v, c, sm, step);
                }
            }
        });
    };

    let t_seam = crate::prof::StageTimer::start("defects.grain.walk.seam");
    // Seam atoms: two or more seam-like bonds.
    let mut seam = vec![false; n];
    if !bonds.offsets.is_empty() {
        par.fill(&mut seam, |k, o| {
            let mut c = 0;
            neighbours(k, &mut |_, _, _, sm, _| c += usize::from(sm));
            *o = c >= 2;
        });
    }
    let n_seam = seam.iter().filter(|&&x| x).count();
    drop(t_seam);

    let mut u: Vec<Coord3> = wrapped.clone();
    // Lattice coordinate (fractional) of each placed atom, relative to its
    // piece's first atom: the sum of the lattice steps of the walked bonds.
    let mut lat_n: Vec<Coord3> = vec![[0.0; 3]; n];
    let mut loops: Vec<([i64; 3], Coord3)> = Vec::new();
    let mut comp = vec![u32::MAX; n];
    let mut sizes: Vec<usize> = Vec::new();
    let mut spans = [false; 3];
    let shift = |spans: &mut [bool; 3], d: Coord3| {
        let f = frac(d);
        for (a, s) in spans.iter_mut().enumerate() {
            if cell.pbc[a] && f[a].round() != 0.0 {
                *s = true;
            }
        }
    };
    // Level-synchronous: each level's neighbour queries run in parallel,
    // reading only finished levels; a newly reached atom takes its position
    // from the lowest-indexed parent, so the layout does not depend on the
    // thread count. Every other bond into a placed atom is a loop check.
    let mut frontier: Vec<u32> = Vec::new();
    for start in 0..n {
        if comp[start] != u32::MAX || seam[start] || !lattice_atom[start] {
            continue;
        }
        let c = sizes.len() as u32;
        sizes.push(0);
        comp[start] = c;
        frontier.clear();
        frontier.push(start as u32);
        while !frontier.is_empty() {
            sizes[c as usize] += frontier.len();
            let (comp_r, u_r, seam_r, n_r) = (&comp, &u, &seam, &lat_n);
            let scan = |lo: usize, hi: usize| {
                let mut reach: Vec<(u32, u32, Coord3, Coord3)> = Vec::new();
                let mut sp = [false; 3];
                let mut lp = Vec::new();
                for &k in &frontier[lo..hi] {
                    let uk = u_r[k as usize];
                    let nk = n_r[k as usize];
                    neighbours(k as usize, &mut |m, v, coherent, _, step| {
                        if !coherent || seam_r[m] || !lattice_atom[m] {
                            return;
                        }
                        let p = [uk[0] + v[0], uk[1] + v[1], uk[2] + v[2]];
                        let np = [nk[0] + step[0], nk[1] + step[1], nk[2] + step[2]];
                        if comp_r[m] == u32::MAX {
                            reach.push((m as u32, k, p, np));
                        } else {
                            let um = u_r[m];
                            let dd = [p[0] - um[0], p[1] - um[1], p[2] - um[2]];
                            shift(&mut sp, dd);
                            if comp_r[m] == c {
                                let nm = n_r[m];
                                let s = frac(dd).map(|x| x.round() as i64);
                                if s != [0; 3] {
                                    lp.push((s, [np[0] - nm[0], np[1] - nm[1], np[2] - nm[2]]));
                                }
                            }
                        }
                    });
                }
                (reach, sp, lp)
            };
            // A level is far smaller than `CHUNK_ATOMS`, so it is split here,
            // into pieces whose results keep their order.
            const PIECE: usize = 256;
            let pieces = crate::exec::pool::ceil_div(frontier.len(), PIECE);
            let parts: Vec<Scan> = if pieces == 1 {
                vec![scan(0, frontier.len())]
            } else {
                let slots: Vec<std::sync::Mutex<Option<_>>> =
                    (0..pieces).map(|_| std::sync::Mutex::new(None)).collect();
                par.for_each(pieces, |q| {
                    let r = scan(q * PIECE, ((q + 1) * PIECE).min(frontier.len()));
                    *slots[q].lock().unwrap() = Some(r);
                });
                slots
                    .into_iter()
                    .map(|m| m.into_inner().unwrap().unwrap())
                    .collect()
            };
            let mut reach: Vec<(u32, u32, Coord3, Coord3)> = Vec::new();
            for (r, sp, lp) in parts {
                reach.extend(r);
                loops.extend(lp);
                for d in 0..3 {
                    spans[d] |= sp[d];
                }
            }
            reach.sort_unstable_by_key(|e| (e.0, e.1));
            frontier.clear();
            for (idx, &(m, _, p, np)) in reach.iter().enumerate() {
                if idx > 0 && reach[idx - 1].0 == m {
                    // Another parent's route to an atom placed this level.
                    let um = u[m as usize];
                    let dd = [p[0] - um[0], p[1] - um[1], p[2] - um[2]];
                    shift(&mut spans, dd);
                    let s = frac(dd).map(|x| x.round() as i64);
                    if s != [0; 3] {
                        let nm = lat_n[m as usize];
                        loops.push((s, [np[0] - nm[0], np[1] - nm[1], np[2] - nm[2]]));
                    }
                    continue;
                }
                comp[m as usize] = c;
                u[m as usize] = p;
                lat_n[m as usize] = np;
                frontier.push(m);
            }
        }
    }
    let _t_attach = crate::prof::StageTimer::start("defects.grain.walk.attach");
    // Seam atoms join the side they bond to best, a layer at a time: each
    // layer reads only atoms placed before it, so the result does not depend
    // on the thread count.
    loop {
        let pending: Vec<u32> = (0..n as u32)
            .filter(|&k| comp[k as usize] == u32::MAX)
            .collect();
        let mut found: Vec<Option<(usize, Coord3)>> = vec![None; pending.len()];
        let (comp_r, u_r) = (&comp, &u);
        par.fill(&mut found, |q, o| {
            let mut best: Option<(f64, usize, Coord3)> = None;
            neighbours(pending[q] as usize, &mut |m, v, coherent, _, _| {
                if coherent && comp_r[m] != u32::MAX {
                    let d = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
                    let um = u_r[m];
                    let p = [um[0] - v[0], um[1] - v[1], um[2] - v[2]];
                    // Shortest bond, ties by index: the nearest placed atom.
                    if best.map_or(true, |b| (d, m) < (b.0, b.1)) {
                        best = Some((d, m, p));
                    }
                }
            });
            *o = best.map(|b| (b.1, b.2));
        });
        let mut placed = 0;
        for (&k, f) in pending.iter().zip(found) {
            if let Some((m, p)) = f {
                comp[k as usize] = comp[m];
                sizes[comp[m] as usize] += 1;
                u[k as usize] = p;
                placed += 1;
            }
        }
        if placed == 0 {
            break;
        }
    }
    // Anything left (a seam atom with no coherent bond) is a piece alone.
    for c in comp.iter_mut() {
        if *c == u32::MAX {
            *c = sizes.len() as u32;
            sizes.push(1);
        }
    }
    // Pieces to the image nearest the largest piece; spanned axes back into
    // the box.
    let main = (0..sizes.len())
        .max_by_key(|&c| (sizes[c], usize::MAX - c))
        .unwrap_or(0);
    let mut centre = vec![[0.0f64; 3]; sizes.len()];
    for k in 0..n {
        let l = cell.x2lamda_pre(u[k], &inv_t);
        let c = comp[k] as usize;
        for (e, v) in centre[c].iter_mut().zip(l) {
            *e += v / sizes[c] as f64;
        }
    }
    let local = (0..n)
        .map(|k| {
            let mut l = cell.x2lamda_pre(u[k], &inv_t);
            let c = comp[k] as usize;
            for d in 0..3 {
                if !cell.pbc[d] {
                    continue;
                }
                if spans[d] {
                    l[d] -= l[d].floor();
                } else {
                    l[d] -= (centre[c][d] - centre[main][d]).round();
                }
            }
            cell.lamda2x(l)
        })
        .collect();
    (local, spans, n_seam, loop_candidates(&loops))
}

/// Each axis's box-vector lattice translations from the walk's loops that go
/// once round the box along it alone, with how many loops gave each: most
/// votes first, ties to the lexicographically smallest translation. Loops on
/// either side of a dislocation differ by its Burgers vector, so an axis with
/// more than one candidate holds one.
fn loop_candidates(loops: &[([i64; 3], Coord3)]) -> [Vec<(Coord3, usize)>; 3] {
    let mut votes: [Vec<(Coord3, usize)>; 3] = Default::default();
    for (s, dn) in loops {
        let nz: Vec<usize> = (0..3).filter(|&d| s[d] != 0).collect();
        let [d] = nz[..] else { continue };
        if s[d].abs() != 1 {
            continue;
        }
        let n = dn.map(|x| x * s[d] as f64);
        match votes[d]
            .iter_mut()
            .find(|(o, _)| (0..3).all(|k| (o[k] - n[k]).abs() <= 1e-6))
        {
            Some(v) => v.1 += 1,
            None => votes[d].push((n, 1)),
        }
    }
    for v in votes.iter_mut() {
        v.sort_by(|a, b| {
            b.1.cmp(&a.1).then_with(|| {
                a.0.iter()
                    .zip(&b.0)
                    .map(|(x, y)| x.total_cmp(y))
                    .find(|o| o.is_ne())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
        });
    }
    votes
}

/// For a grain that is the whole periodic box, where loops round the box
/// disagree (a dislocation spans it): the candidate translations whose
/// lattice holds as many sites as there are atoms, `|det N| * basis` sites,
/// closest to `n_atoms`; ties to the most votes. A box-spanning dislocation
/// bounds a strip of extra or missing plane, so going round the box on one
/// side of it crosses one plane more than on the other. The majority side is
/// whichever the strip covers most of, not the crystal the atoms were cut
/// from: counting against it books a whole plane (thousands of sites) as
/// vacancies or interstitials, and V - I is then off by exactly one plane.
/// The crystal that conserves the atoms is the one they were cut from
/// (atoms are not created or lost in a periodic box). Returns the
/// translations and their lattice's sites minus `n_atoms`; `None` where any
/// axis has no candidate.
fn conserving_translations(
    cands: &[Vec<(Coord3, usize)>; 3],
    n_basis: usize,
    n_atoms: usize,
) -> Option<([Coord3; 3], i64)> {
    const TOP: usize = 4;
    let mut best: Option<(i64, usize, [Coord3; 3])> = None;
    for a in cands[0].iter().take(TOP) {
        for b in cands[1].iter().take(TOP) {
            for c in cands[2].iter().take(TOP) {
                let det = Mat3 { m: [a.0, b.0, c.0] }.det().abs();
                let diff = (det * n_basis as f64).round() as i64 - n_atoms as i64;
                let votes = a.1 + b.1 + c.1;
                let better = best.map_or(true, |(d, v, _)| {
                    diff.abs() < d.abs() || (diff.abs() == d.abs() && votes > v)
                });
                if better {
                    best = Some((diff, votes, [a.0, b.0, c.0]));
                }
            }
        }
    }
    best.map(|b| (b.2, b.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// BCC first shell, a = 3.165 Å, in the cube frame.
    fn bcc_bonds(offsets: Vec<Coord3>) -> Bonds {
        let h = 3.165 / 2.0;
        let mut shell = Vec::new();
        for sx in [-h, h] {
            for sy in [-h, h] {
                for sz in [-h, h] {
                    shell.push([sx, sy, sz]);
                }
            }
        }
        let nn = 3.165 * 3f64.sqrt() / 2.0;
        let shell_frac = shell.iter().map(|v| v.map(|x| x / 3.165)).collect();
        Bonds {
            shell,
            shell_frac,
            offsets,
            nn,
            coherent: 0.1 * nn,
        }
    }

    #[test]
    fn a_bond_off_the_lattice_toward_no_offset_is_not_walked() {
        let b = bcc_bonds(vec![[-0.83, 0.36, -0.08], [0.83, -0.36, 0.08]]);
        let s = b.shell[0];
        let along = |d: Coord3| [s[0] + d[0], s[1] + d[1], s[2] + d[2]];
        // Thermal: walked.
        assert_eq!(b.read(along([0.02, -0.01, 0.03])), (true, false));
        // Across the seam: seam-like, not walked.
        assert_eq!(b.read(along([-0.82, 0.37, -0.07])), (false, true));
        // An atom left 0.45 Å out of place on an atomsk seam, toward no closure
        // offset: a 0.35-bond tolerance would walk it, which lets the walk
        // cross the seam and refuse the grain. It is neither.
        assert_eq!(b.read(along([-0.11, 0.42, 0.11])), (false, false));
    }

    #[test]
    fn loops_round_the_box_vote_and_flag_a_dislocation() {
        let n = [33.0, 31.0, -26.0];
        let slipped = [33.5, 30.5, -26.5]; // n + a/2[1 -1 1]: a Burgers vector
        let mut loops = vec![([0, 1, 0], n); 5];
        loops.push(([0, -1, 0], n.map(|x| -x))); // the same loop walked backwards
        loops.extend(vec![([0, 1, 0], slipped); 2]);
        loops.push(([1, 1, 0], [9.0, 9.0, 9.0])); // diagonal: no single axis
        loops.push(([0, 0, 2], [1.0, 2.0, 3.0])); // twice round: not used
        let c = loop_candidates(&loops);
        assert!(c[0].is_empty() && c[2].is_empty());
        assert_eq!(c[1], vec![(n, 6), (slipped, 2)]);

        let c = loop_candidates(&[([-1, 0, 0], [-4.5, -1.5, 2.5])]);
        assert_eq!(c[0], vec![([4.5, 1.5, -2.5], 1)]);
    }

    #[test]
    fn a_box_spanning_dislocation_snaps_to_the_lattice_that_holds_the_atoms() {
        // FCC, 4 sites per cell, 50 cells a side: 500000 sites. Most x loops
        // run through a strip of extra plane and count 51 cells; the
        // minority's 50 holds the atoms.
        let x = vec![([51.0, 0.0, 0.0], 9), ([50.0, 0.0, 0.0], 3)];
        let y = vec![([0.0, 50.0, 0.0], 12)];
        let z = vec![([0.0, 0.0, 50.0], 12)];
        let (t, d) =
            conserving_translations(&[x.clone(), y.clone(), z.clone()], 4, 500_000).unwrap();
        assert_eq!((t[0], d), ([50.0, 0.0, 0.0], 0));
        // Atoms no candidate holds: the nearest, and the shortfall reported.
        let (t, d) = conserving_translations(&[x, y.clone(), z.clone()], 4, 520_000).unwrap();
        assert_eq!((t[0], d), ([51.0, 0.0, 0.0], -10_000));
        // An axis without loops: nothing to choose.
        assert!(conserving_translations(&[vec![], y, z], 4, 500_000).is_none());
    }
}
